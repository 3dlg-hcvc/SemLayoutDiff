import os
import moderngl

# Set the backend to EGL
os.environ['MODERNGL_BACKEND'] = 'egl'

# Monkey patch the context creation in BaseScene
original_create_context = moderngl.create_standalone_context

def create_egl_context(*args, **kwargs):
    return original_create_context(backend='egl')

moderngl.create_standalone_context = create_egl_context