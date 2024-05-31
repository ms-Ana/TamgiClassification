import glob
import os

import bpy

f = "/home/ana/University/Tamgi/data/util_objects/AncientVase/Ancient_Vase.obj"
bpy.ops.import_scene.obj(filepath=f)
mat = bpy.context.active_object.material_slots[0].material

imgpath = "/home/ana/University/Tamgi/data/textures/text2.jpg"
img = bpy.data.images.load(imgpath)
file_path = "/home/ana/University/Tamgi/data/util_objects/Pottery/Pottery.blend"
inner_path = "Object"
object_name = "Pottery"
bpy.ops.wm.append(
    filepath=os.path.join(file_path, inner_path, object_name),
    directory=os.path.join(file_path, inner_path),
    filename=object_name,
)

import os

import bpy

imgpath = "/home/ana/University/Tamgi/data/textures/text2.jpg"
img = bpy.data.images.load(imgpath)
file_path = "/home/ana/University/Tamgi/data/util_objects/Pottery/Pottery.blend"
inner_path = "Object"
object_name = "Jarron3"
bpy.ops.wm.append(
    filepath=os.path.join(file_path, inner_path, object_name),
    directory=os.path.join(file_path, inner_path),
    filename=object_name,
)
