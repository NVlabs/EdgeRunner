'''
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import os
import cv2
import tqdm
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

GUI_AVAILABLE = True
try:
    import dearpygui.dearpygui as dpg
except Exception as e:
    print('[WARN] cannot import dearpygui, assume running with --wogui')
    GUI_AVAILABLE = False

import kiui
from kiui.mesh import Mesh
from kiui.cam import OrbitCamera
from kiui.op import safe_normalize

from meto import Engine, load_mesh, sort_mesh, normalize_mesh

class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        if not GUI_AVAILABLE and not opt.wogui:
            print(f'[WARN] cannot import dearpygui, assume running with --wogui')
        self.wogui = not GUI_AVAILABLE or opt.wogui # disable gui and run in cmd
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.bg_color = torch.ones(3, dtype=torch.float32).cuda() # default white bg
        # self.bg_color = torch.zeros(3, dtype=torch.float32).cuda() # black bg

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.light_dir = np.array([0, 0])
        self.ambient_ratio = 0.5

        # auto-rotate
        self.auto_rotate_cam = False
        self.auto_rotate_light = False

        # engine
        self.engine = Engine(opt.discrete_bins, verbose=opt.verbose, backend=opt.backend)
        
        # load mesh or tokens
        if opt.mesh is not None:
            vertices, faces = load_mesh(opt.mesh, clean=True)
            vertices = normalize_mesh(vertices)
            self.mesh = Mesh(v=torch.from_numpy(vertices).float().cuda(), f=torch.from_numpy(faces).int().cuda(), device='cuda')
            self.mesh.auto_normal()

            tokens, face_order, face_type = self.engine.encode(vertices, faces)
            print(f'[INFO] compressed {faces.shape[0]} to {len(tokens)} tokens, ratio = {100 * len(tokens) / (9 * faces.shape[0]):.2f} %')
            kiui.lo(tokens)

            # default is to visualize encoding process
            if not opt.decode:
                self.face_order = torch.from_numpy(face_order).cuda()
            else:
                # to visualize decoding process
                vertices, faces, face_type = self.engine.decode(tokens)
                self.mesh = Mesh(v = torch.from_numpy(vertices).float().cuda(), f = torch.from_numpy(faces).int().cuda(), device='cuda')
                self.mesh.auto_normal()
                # dummy order
                self.face_order = torch.arange(len(faces), dtype=torch.int32).cuda()

        elif opt.tokens is not None:
            # decode mesh from tokens
            # NOTE: make sure discrete_bins and backend are matched, otherwise segmentation fault
            tokens = np.load(opt.tokens)
            kiui.lo(tokens)
            vertices, faces, face_type = self.engine.decode(tokens)
            self.mesh = Mesh(v = torch.from_numpy(vertices).float().cuda(), f = torch.from_numpy(faces).int().cuda(), device='cuda')
            self.mesh.auto_normal()
            # dummy order
            self.face_order = torch.arange(len(faces), dtype=torch.int32).cuda()

        else:
            raise ValueError('either --mesh or --tokens should be provided')
        
        face_type = np.concatenate([np.zeros(1, dtype=np.int32), face_type + 1], axis=0) # 0 for background, offset face index by 1
        self.face_type = torch.from_numpy(face_type).cuda()
        
        # if opt.backend == 'LR_ABSCO':
        #     self.face_color_mapping = torch.tensor([
        #         [255, 255, 255], # background = white
        #         [255, 242, 204], # OP_L 
        #         [226, 240, 217], # OP_R 
        #         [222, 235, 247], # OP_E 
        #     ], dtype=torch.float32).cuda() / 255
        # else:
        #     self.face_color_mapping = torch.tensor([
        #         [255, 255, 255], # background = white
        #         [246, 214, 214], # OP_C 
        #         [255, 242, 204], # OP_L 
        #         [222, 235, 247], # OP_E 
        #         [226, 240, 217], # OP_R 
        #         [224, 193, 255], # OP_S 
        #     ], dtype=torch.float32).cuda() / 255
        self.face_color_mapping = torch.tensor([
            [255, 255, 255], # background = white
            [255, 89, 94], # OP_C = red
            [138, 201, 38], # OP_L = green
            [25, 130, 196], # OP_E = blue
            [255, 202, 58], # OP_R = yellow
            [106, 76, 147], # OP_S = purple
        ], dtype=torch.float32).cuda() / 255

        # current number of faces to show
        self.num_visible_faces = self.face_order.shape[0]
        self.invert_visible_faces = False

        # render_mode
        self.render_modes = ['face_type', 'depth', 'normal']
        if self.mesh.albedo is not None or self.mesh.vc is not None:
            self.render_modes.extend(['albedo', 'lambertian'])
        
        if opt.mode in self.render_modes:
            self.mode = opt.mode
        else:
            print(f'[WARN] mode {opt.mode} not supported, fallback to render normal')
            self.mode = 'normal' # fallback

        # display wireframe
        self.show_wire = True

        if not opt.force_cuda_rast and (self.wogui or os.name == 'nt'):
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

        if not self.wogui:
            dpg.create_context()
            self.register_dpg()
            self.step()
        

    def __del__(self):
        if not self.wogui:
            dpg.destroy_context()
    
    def step(self):

        if not self.need_update:
            return
    
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # do MVP for vertices
        pose = torch.from_numpy(self.cam.pose.astype(np.float32)).cuda()
        proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
        
        v_cam = torch.matmul(F.pad(self.mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        # limit visible faces
        if self.invert_visible_faces:
            visible_ids = self.face_order[min(self.num_visible_faces, len(self.face_order) - 1):]
        else:
            visible_ids = self.face_order[:self.num_visible_faces]

        f = self.mesh.f[visible_ids]
        ft = self.mesh.ft[visible_ids] if self.mesh.ft is not None else None
        fn = self.mesh.fn[visible_ids] if self.mesh.fn is not None else None

        H, W = int(self.opt.ssaa * self.H), int(self.opt.ssaa * self.W)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (H, W))

        alpha = (rast[..., 3:] > 0).float()
        alpha = dr.antialias(alpha, rast, v_clip, f).squeeze(0).clamp(0, 1) # [H, W, 3]
        
        if self.mode == 'depth':
            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, f) # [1, H, W, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            buffer = depth.squeeze(0).detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
        elif self.mode == 'normal':
            normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, fn)
            normal = safe_normalize(normal)
            normal_image = (normal[0] + 1) / 2
            normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1).to(normal_image.device)) # remove background
            buffer = normal_image[0].detach().cpu().numpy()
        elif self.mode == 'face_type':
            face_ids = rast[..., -1].long() # [1, H, W]
            buffer_id = self.face_type[face_ids] # [1, H, W]
            buffer = self.face_color_mapping[buffer_id] # [1, H, W, 3]
            buffer = buffer[0].detach().cpu().numpy()
        else:
            # use vertex color if exists
            if self.mesh.vc is not None:
                albedo, _ = dr.interpolate(self.mesh.vc.unsqueeze(0).contiguous(), rast, f)
            # use texture image
            else: # assert mesh.albedo is not None
                texc, _ = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, ft)
                albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]

            albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)) # remove background
            # albedo = dr.antialias(albedo, rast, v_clip, f).clamp(0, 1) # [1, H, W, 3]

            if self.mode == 'albedo':
                albedo = albedo * alpha + self.bg_color * (1 - alpha)
                buffer = albedo[0].detach().cpu().numpy()
            else:
                normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, fn)
                normal = safe_normalize(normal)
                
                light_d = np.deg2rad(self.light_dir)
                light_d = np.array([
                    np.cos(light_d[0]) * np.sin(light_d[1]),
                    -np.sin(light_d[0]),
                    np.cos(light_d[0]) * np.cos(light_d[1]),
                ], dtype=np.float32)
                light_d = torch.from_numpy(light_d).to(albedo.device)
                lambertian = self.ambient_ratio + (1 - self.ambient_ratio)  * (normal @ light_d).float().clamp(min=0)
                albedo = (albedo * lambertian.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
                buffer = albedo[0].detach().cpu().numpy()
                    

        if self.show_wire:
            u = rast[..., 0] # [1, h, w]
            v = rast[..., 1] # [1, h, w]
            w = 1 - u - v
            mask = rast[..., 2]

            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, f) # [1, H, W, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            adaptive_width = 0.01 + depth.squeeze(-1) ** 2 * 0.02

            near_edge = (((w < adaptive_width) | (u < adaptive_width) | (v < adaptive_width)) & (mask > 0))[0].detach().cpu().numpy() # [h, w]
            buffer[near_edge] = np.array([0, 0, 0], dtype=np.float32) # black wire
        
        # ssaa rescale
        if H != self.H or W != self.W:
            buffer = cv2.resize(buffer, (self.H, self.W), interpolation=cv2.INTER_AREA)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.render_buffer = buffer
        self.need_update = False

        if self.auto_rotate_cam:
            self.cam.orbit(5, 0)
            self.need_update = True
        
        if self.auto_rotate_light:
            self.light_dir[1] += 3
            self.need_update = True
        
        if not self.wogui:
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=200, collapsed=True):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)              

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(self.render_modes, label='mode', default_value=self.mode, tag="_mode_combo", callback=callback_change_mode)

                # show wireframe
                def callback_toggle_wireframe(sender, app_data):
                    self.show_wire = not self.show_wire
                    dpg.set_value("_checkbox_wire", self.show_wire)
                    self.need_update = True

                dpg.add_checkbox(label="wireframe", tag="_checkbox_wire", default_value=self.show_wire, callback=callback_toggle_wireframe)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32).cuda() # only need RGB in [0, 1]
                    self.need_update = True
                
                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(label="FoVY", min_value=1, max_value=120, format="%d deg", default_value=np.rad2deg(self.cam.fovy), callback=callback_set_fovy)

                # num_visible_faces slider
                def callback_set_num_faces(sender, app_data):
                    self.num_visible_faces = app_data
                    self.need_update = True

                dpg.add_slider_int(label="Progress", tag='_slider_num_faces', min_value=1, max_value=len(self.face_order), default_value=self.num_visible_faces, callback=callback_set_num_faces)

                # invert visible faces
                def callback_toggle_invert_faces(sender, app_data):
                    self.invert_visible_faces = not self.invert_visible_faces
                    dpg.set_value("_checkbox_invert_faces", self.invert_visible_faces)
                    self.need_update = True

                dpg.add_checkbox(label="Invert Faces", tag='_checkbox_invert_faces', default_value=self.invert_visible_faces, callback=callback_toggle_invert_faces)

                # light dir
                def callback_set_light_dir(sender, app_data, user_data):
                    self.light_dir[user_data] = app_data
                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Plane Light Direction:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="elevation", min_value=-90, max_value=90, format="%.2f", default_value=self.light_dir[0], callback=callback_set_light_dir, user_data=0)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="azimuth", min_value=0, max_value=360, format="%.2f", default_value=self.light_dir[1], callback=callback_set_light_dir, user_data=1)

                # ambient ratio
                def callback_set_abm_ratio(sender, app_data):
                    self.ambient_ratio = app_data
                    self.need_update = True

                dpg.add_slider_float(label="ambient", min_value=0, max_value=1.0, format="%.5f", default_value=self.ambient_ratio, callback=callback_set_abm_ratio)

        ### register IO handlers

        # camera mouse controller
        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        # press spacebar to toggle rendering mode
        def callback_space_toggle_mode(sender, app_data):
            self.mode = self.render_modes[(self.render_modes.index(self.mode) + 1) % len(self.render_modes)]
            dpg.set_value("_mode_combo", self.mode)
            self.need_update = True
        
        # press P to toggle auto-rotate camera
        def callback_toggle_auto_rotate_cam(sender, app_data):
            self.auto_rotate_cam = not self.auto_rotate_cam
            self.need_update = True
        
        # press L to toggle auto-rotate light
        def callback_toggle_auto_rotate_light(sender, app_data):
            self.auto_rotate_light = not self.auto_rotate_light
            self.need_update = True
        
        # press right arrow to add 1 to num_visible_faces
        def callback_increase_visible_faces(sender, app_data):
            self.num_visible_faces = min(self.num_visible_faces + 1, len(self.face_order))
            dpg.set_value("_slider_num_faces", self.num_visible_faces)
            self.need_update = True
        
        # press left arrow to subtract 1 to num_visible_faces
        def callback_decrease_visible_faces(sender, app_data):
            self.num_visible_faces = max(self.num_visible_faces - 1, 1)
            dpg.set_value("_slider_num_faces", self.num_visible_faces)
            self.need_update = True
    
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=callback_space_toggle_mode)
            dpg.add_key_press_handler(dpg.mvKey_P, callback=callback_toggle_auto_rotate_cam)
            dpg.add_key_press_handler(dpg.mvKey_L, callback=callback_toggle_auto_rotate_light)
            dpg.add_key_press_handler(dpg.mvKey_W, callback=callback_toggle_wireframe)
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_increase_visible_faces)
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_decrease_visible_faces)
            dpg.add_key_press_handler(dpg.mvKey_I, callback=callback_toggle_invert_faces)

        
        dpg.create_viewport(title='mesh viewer', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):
        assert not self.wogui
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default=None, help="path to mesh (obj, ply, glb, ...)")
    parser.add_argument('--tokens', type=str, default=None, help="path to tokens")
    parser.add_argument('--discrete_bins', type=int, default=512, help="number of bins for discrete encoding")
    parser.add_argument('--decode', action='store_true', help="visualize the decoding process")
    parser.add_argument('--backend', type=str, default='LR_ABSCO', help="engine backend")
    parser.add_argument('--verbose', action='store_true', help="print verbose output")
    parser.add_argument('--mode', default='face_type', type=str, choices=['face_type', 'lambertian', 'albedo', 'normal', 'depth'], help="rendering mode")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--ssaa', type=float, default=2, help="super-sampling anti-aliasing ratio")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument("--wogui", action='store_true', help="disable all dpg GUI")
    parser.add_argument("--force_cuda_rast", action='store_true', help="force to use RasterizeCudaContext.")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--num_azimuth', type=int, default=8, help="number of images to render from different azimuths")
    parser.add_argument('--save_video', type=str, default=None, help="path to save rendered video")

    opt = parser.parse_args()

    gui = GUI(opt)

    if opt.save_video is not None:
        import imageio
        images = []
        # total step is based on num_face
        gui.num_visible_faces = 0
        elevation = [opt.elevation,]
        azimuth = np.arange(0, 360, 3, dtype=np.int32) # front-->back-->front
        gui.cam.from_angle(elevation[0], azimuth[0])
        for i in tqdm.trange(len(gui.face_order) + len(azimuth)):
            if i >= len(gui.face_order):
                ele = elevation[(i - len(gui.face_order)) % len(elevation)]
                azi = azimuth[(i - len(gui.face_order)) % len(azimuth)]
                gui.cam.from_angle(ele, azi)
            gui.num_visible_faces = min(gui.num_visible_faces + 1, len(gui.face_order))
            gui.need_update = True
            gui.step()
            if not opt.wogui:
                dpg.render_dearpygui_frame()
            image = (gui.render_buffer * 255).astype(np.uint8)
            images.append(image)

        images = np.stack(images, axis=0)
        
        imageio.mimwrite(opt.save_video, images, fps=60, quality=8, macro_block_size=1)
    else:
        gui.render()


if __name__ == '__main__':
    main()