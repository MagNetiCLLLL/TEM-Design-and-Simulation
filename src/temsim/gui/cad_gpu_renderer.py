"""Retained, full-detail CAD rasterization on a hardware OpenGL device.

This renderer owns an offscreen context rather than a widget.  The caller can
keep its ordinary QWidget, exact CPU selection and software fallback.  CAD
coordinates and camera transforms stay double precision until clip coordinates
have been formed, so a small part is not rounded away by the column's length.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes

import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext, QSurfaceFormat


class GpuRenderingUnavailable(RuntimeError):
    """A usable hardware context could not be created or has stopped working."""


_VERTEX_SHADER = """#version 410 core
layout(location = 0) in dvec3 position;
layout(location = 1) in dvec3 normal;
layout(location = 2) in vec3 base;
layout(location = 3) in int face_id;
layout(location = 4) in int selected;
uniform dmat3 rotation;
uniform dvec3 centre;
uniform dvec2 scale;
uniform dvec2 offset;
uniform ivec2 viewport;
uniform double depth_extent;
flat out vec3 colour;
flat out int triangle_id;
flat out dvec3 plane_normal;
flat out double plane_offset;
void main() {
    dvec3 camera = rotation * (position - centre);
    dvec2 screen = dvec2(camera.x * scale.x, -camera.y * scale.y) + offset;
    gl_Position = vec4(float(2.0lf * screen.x / double(viewport.x) - 1.0lf),
                       float(1.0lf - 2.0lf * screen.y / double(viewport.y)),
                       float(-camera.z / depth_extent), 1.0);
    vec3 light = normalize(vec3(-0.3, 0.45, 0.84));
    plane_normal = rotation * normal;
    plane_offset = dot(plane_normal, camera);
    float shade = 0.38 + 0.62 * abs(dot(vec3(plane_normal), light));
    vec3 tint = selected != 0 ? 0.35 * base + 0.65 * vec3(255, 196, 84) : base;
    colour = floor(clamp(shade * tint, 0.0, 255.0)) / 255.0;
    triangle_id = face_id;
}
"""

_FRAGMENT_SHADER = """#version 410 core
flat in vec3 colour;
flat in int triangle_id;
flat in dvec3 plane_normal;
flat in double plane_offset;
uniform dvec2 scale;
uniform dvec2 offset;
uniform ivec2 viewport;
uniform double depth_extent;
layout(location = 0) out vec4 rgba;
layout(location = 1) out int identity;
void main() {
    if (plane_normal.z == 0.0lf) discard;
    rgba = vec4(colour, 1.0);
    identity = triangle_id;
    // Compute the depth from the double-precision face plane. Hardware depth
    // interpolation otherwise depends on each triangle's diagonal, allowing
    // two coincident CAD surfaces to fight even with a strict LESS test.
    double x = (double(gl_FragCoord.x) - offset.x) / scale.x;
    double y = (offset.y - (double(viewport.y) - double(gl_FragCoord.y))) / scale.y;
    double z = (plane_offset - plane_normal.x * x - plane_normal.y * y) / plane_normal.z;
    gl_FragDepth = float((1.0lf - z / depth_extent) * 0.5lf);
}
"""


class CadGpuRenderer:
    """Hardware renderer whose geometry survives camera/viewport changes.

    Construct and use on the Qt GUI thread.  All failures are surfaced to the
    caller, which should retain the existing software renderer.  ``render``
    returns top-down RGBA and original rendered-face IDs; background IDs are -1.
    No CAD simplification, back-face culling or mesh welding is performed.
    """

    def __init__(self):
        self._context = None
        self._surface = None
        self._gl = None
        self._program = 0
        self._vao = 0
        self._buffers = []
        self._fbo = 0
        self._textures = []
        self._frame_size = (0, 0)
        self._vertex_count = 0
        self._face_ids = np.empty(0, dtype=np.int32)
        self._selected_faces = None
        self._closed = False
        self.renderer_name = ""
        self.geometry_uploads = 0
        self.rendered_frames = 0
        app = QGuiApplication.instance()
        if app is None or QThread.currentThread() != app.thread():
            raise GpuRenderingUnavailable("CAD OpenGL rendering requires the Qt GUI thread")
        try:
            # Import lazily: missing/broken graphics drivers must not prevent
            # importing the ordinary CAD view and its CPU fallback.
            from OpenGL import GL
            from OpenGL.GL.shaders import compileProgram, compileShader

            self._gl = GL
            requested = QSurfaceFormat()
            requested.setVersion(4, 1)
            requested.setProfile(QSurfaceFormat.CoreProfile)
            requested.setDepthBufferSize(32)
            requested.setSamples(0)
            context = QOpenGLContext()
            self._context = context
            context.setFormat(requested)
            if not context.create():
                raise GpuRenderingUnavailable("No OpenGL context is available")
            actual = context.format()
            if (actual.majorVersion(), actual.minorVersion()) < (4, 1):
                raise GpuRenderingUnavailable("CAD GPU rendering requires OpenGL 4.1")
            surface = QOffscreenSurface()
            self._surface = surface
            surface.setFormat(actual)
            surface.create()
            if not surface.isValid():
                raise GpuRenderingUnavailable("No offscreen OpenGL surface is available")
            with self._current():
                self.renderer_name = (GL.glGetString(GL.GL_RENDERER) or b"").decode(
                    "utf-8", errors="replace"
                )
                name = self.renderer_name.lower()
                if not name or any(token in name for token in (
                    "llvmpipe", "softpipe", "swiftshader", "software", "gdi generic",
                    "microsoft basic render", "mesa offscreen",
                )):
                    raise GpuRenderingUnavailable(
                        "No hardware OpenGL renderer is available: " + (self.renderer_name or "unknown")
                    )
                self._program = int(compileProgram(
                    compileShader(_VERTEX_SHADER, GL.GL_VERTEX_SHADER),
                    compileShader(_FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER),
                ))
                self._uniforms = {
                    key: GL.glGetUniformLocation(self._program, key)
                    for key in ("rotation", "centre", "scale", "offset", "viewport", "depth_extent")
                }
                self._vao = int(GL.glGenVertexArrays(1))
                self._buffers = [int(value) for value in GL.glGenBuffers(5)]
        except Exception as exc:
            self.close()
            if isinstance(exc, GpuRenderingUnavailable):
                raise
            raise GpuRenderingUnavailable(f"CAD OpenGL initialization failed: {exc}") from exc

    @contextmanager
    def _current(self):
        if self._closed or self._context is None or self._surface is None:
            raise GpuRenderingUnavailable("CAD OpenGL renderer is closed")
        if QThread.currentThread() != self._context.thread():
            raise GpuRenderingUnavailable("CAD OpenGL rendering must stay on its Qt GUI thread")
        previous = QOpenGLContext.currentContext()
        previous_surface = previous.surface() if previous is not None else None
        if not self._context.makeCurrent(self._surface):
            raise GpuRenderingUnavailable("CAD OpenGL context is no longer available")
        try:
            yield
        finally:
            self._context.doneCurrent()
            if previous is not None and previous is not self._context and previous_surface is not None:
                previous.makeCurrent(previous_surface)

    def set_geometry(self, triangles, normals, bases, face_ids):
        """Replace complete prepared triangles, uploading them once to the GPU."""
        triangles = np.asarray(triangles, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        bases = np.asarray(bases, dtype=np.float32)
        face_ids = np.asarray(face_ids, dtype=np.int32)
        count = len(triangles)
        if (
            triangles.shape != (count, 3, 3)
            or normals.shape != (count, 3)
            or bases.shape != (count, 3)
            or face_ids.shape != (count,)
            or not np.isfinite(triangles).all()
            or not np.isfinite(normals).all()
            or not np.isfinite(bases).all()
            or np.any(face_ids < 0)
        ):
            raise ValueError("CAD GPU geometry must contain finite triangles, face normals, colours and IDs")
        GL = self._gl
        arrays = (
            np.ascontiguousarray(triangles.reshape(-1, 3)),
            np.ascontiguousarray(np.repeat(normals, 3, axis=0)),
            np.ascontiguousarray(np.repeat(bases, 3, axis=0)),
            np.ascontiguousarray(np.repeat(face_ids, 3)),
            np.zeros(count * 3, dtype=np.uint8),
        )
        with self._current():
            GL.glBindVertexArray(self._vao)
            try:
                for location, (buffer, array) in enumerate(zip(self._buffers, arrays)):
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer)
                    GL.glBufferData(GL.GL_ARRAY_BUFFER, array.nbytes, array,
                                    GL.GL_DYNAMIC_DRAW if location == 4 else GL.GL_STATIC_DRAW)
                    GL.glEnableVertexAttribArray(location)
                    if location <= 1:
                        GL.glVertexAttribLPointer(location, 3, GL.GL_DOUBLE, 0, ctypes.c_void_p(0))
                    elif location < 3:
                        GL.glVertexAttribPointer(location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
                    else:
                        GL.glVertexAttribIPointer(location, 1,
                                                  GL.GL_INT if location == 3 else GL.GL_UNSIGNED_BYTE,
                                                  0, ctypes.c_void_p(0))
            finally:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glBindVertexArray(0)
        self._vertex_count = count * 3
        self._face_ids = np.array(face_ids, copy=True)
        self._selected_faces = np.empty(0, dtype=np.int32)
        self.geometry_uploads += 1

    def _ensure_framebuffer(self, width, height):
        if self._frame_size == (width, height):
            return
        GL = self._gl
        if self._fbo:
            GL.glDeleteFramebuffers(1, [self._fbo])
            self._fbo = 0
        if self._textures:
            GL.glDeleteTextures(self._textures)
            self._textures = []
        self._fbo = int(GL.glGenFramebuffers(1))
        self._textures = [int(value) for value in GL.glGenTextures(3)]
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        for texture, internal, external, kind, attachment in (
            (self._textures[0], GL.GL_RGBA8, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, GL.GL_COLOR_ATTACHMENT0),
            (self._textures[1], GL.GL_R32I, GL.GL_RED_INTEGER, GL.GL_INT, GL.GL_COLOR_ATTACHMENT1),
            (self._textures[2], GL.GL_DEPTH_COMPONENT32F, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, GL.GL_DEPTH_ATTACHMENT),
        ):
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, width, height, 0, external, kind, None)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, attachment, GL.GL_TEXTURE_2D, texture, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDrawBuffers(2, [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1])
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise GpuRenderingUnavailable("CAD OpenGL framebuffer is incomplete")
        self._frame_size = (width, height)

    def render(self, width, height, rotation, center, scale, offset, depth_extent, selected_faces=()):
        """Render using pixel scale/offset and symmetric camera-depth bounds.

        ``scale`` is (horizontal, vertical) pixels per mm. ``offset`` is the
        projected origin in top-down pixels. ``depth_extent`` must enclose all
        visible geometry relative to ``center``; larger camera Z is nearer.
        """
        width, height = int(width), int(height)
        rotation = np.ascontiguousarray(rotation, dtype=np.float64)
        center = np.ascontiguousarray(center, dtype=np.float64)
        scale = np.ascontiguousarray(scale, dtype=np.float64)
        offset = np.ascontiguousarray(offset, dtype=np.float64)
        selected = np.unique(np.asarray(selected_faces, dtype=np.int32))
        if (width < 1 or height < 1 or rotation.shape != (3, 3) or center.shape != (3,)
                or scale.shape != (2,) or offset.shape != (2,)
                or not all(np.isfinite(value).all() for value in (rotation, center, scale, offset))
                or not np.isfinite(depth_extent) or depth_extent <= 0):
            raise ValueError("CAD GPU camera dimensions and transform must be finite and valid")
        GL = self._gl
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        ids = np.empty((height, width), dtype=np.int32)
        with self._current():
            try:
                self._ensure_framebuffer(width, height)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
                GL.glViewport(0, 0, width, height)
                for capability in (GL.GL_BLEND, GL.GL_CULL_FACE, GL.GL_DITHER, GL.GL_MULTISAMPLE,
                                   GL.GL_SCISSOR_TEST, GL.GL_FRAMEBUFFER_SRGB):
                    GL.glDisable(capability)
                GL.glEnable(GL.GL_DEPTH_TEST)
                GL.glDepthMask(GL.GL_TRUE)
                GL.glDepthFunc(GL.GL_LESS)
                GL.glClearBufferfv(GL.GL_COLOR, 0, np.array([18, 25, 36, 255], dtype=np.float32) / 255)
                GL.glClearBufferiv(GL.GL_COLOR, 1, np.full(4, -1, dtype=np.int32))
                GL.glClearBufferfv(GL.GL_DEPTH, 0, np.array([1], dtype=np.float32))
                GL.glUseProgram(self._program)
                uniforms = self._uniforms
                GL.glUniformMatrix3dv(uniforms["rotation"], 1, GL.GL_TRUE, rotation)
                GL.glUniform3dv(uniforms["centre"], 1, center)
                GL.glUniform2dv(uniforms["scale"], 1, scale)
                GL.glUniform2dv(uniforms["offset"], 1, offset)
                GL.glUniform2i(uniforms["viewport"], width, height)
                GL.glUniform1d(uniforms["depth_extent"], float(depth_extent))
                if self._selected_faces is None or not np.array_equal(selected, self._selected_faces):
                    mask = np.repeat(np.isin(self._face_ids, selected).astype(np.uint8), 3)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffers[4])
                    GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, mask.nbytes, mask)
                    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                    self._selected_faces = selected.copy()
                GL.glBindVertexArray(self._vao)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._vertex_count)
                GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
                GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
                GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels)
                GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
                GL.glReadPixels(0, 0, width, height, GL.GL_RED_INTEGER, GL.GL_INT, ids)
            finally:
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.rendered_frames += 1
        return np.ascontiguousarray(pixels[::-1]), np.ascontiguousarray(ids[::-1])

    def close(self):
        """Release this renderer without disturbing another view's GL context."""
        if self._closed:
            return
        try:
            if self._context is not None and self._surface is not None and self._gl is not None:
                with self._current():
                    GL = self._gl
                    if self._fbo:
                        GL.glDeleteFramebuffers(1, [self._fbo])
                    if self._textures:
                        GL.glDeleteTextures(self._textures)
                    if self._buffers:
                        GL.glDeleteBuffers(len(self._buffers), self._buffers)
                    if self._vao:
                        GL.glDeleteVertexArrays(1, [self._vao])
                    if self._program:
                        GL.glDeleteProgram(self._program)
        except Exception:
            # Context loss is why callers may be falling back. The driver owns
            # remaining objects until that context is destroyed.
            pass
        finally:
            self._closed = True
            if self._surface is not None:
                try:
                    self._surface.destroy()
                except RuntimeError:
                    pass
            self._surface = None
            self._context = None
            self._buffers = []
            self._textures = []
