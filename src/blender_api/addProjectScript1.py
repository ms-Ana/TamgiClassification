import bpy
from bpy.types import Modifier


# Clear all nodes in a mat
def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


# Очистить всё
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# создаем новую пр=лоскость

bpy.ops.mesh.primitive_plane_add(
    size=20,
    calc_uvs=True,
    enter_editmode=False,
    align="WORLD",
    location=(0, 0, 0),
    rotation=(0, 0, 0),
    scale=(0, 0, 0),
)
current_name = bpy.context.selected_objects[0].name
plane = bpy.data.objects[current_name]
plane.name = "myPlane"

mod = plane.modifiers.new("Subdivision", "SUBSURF")
mod.subdivision_type = "SIMPLE"
bpy.context.object.cycles.use_adaptive_subdivision = True
bpy.context.object.cycles.dicing_rate = 0.5


# получить материал
mat = bpy.data.materials.get("Material")
if mat is None:
    # создать материал
    mat = bpy.data.materials.new(name="Material")


clear_material(mat)
mat.use_nodes = True

nodes = mat.node_tree.nodes
links = mat.node_tree.links


output = nodes.new(type="ShaderNodeOutputMaterial")
output.location = (1500.0, 100.0)
mix = nodes.new(type="ShaderNodeMixShader")
mix.location = (1300.0, 300.0)
principled_1 = nodes.new(type="ShaderNodeBsdfPrincipled")
principled_1.location = (700.0, 300.0)
principled_2 = nodes.new(type="ShaderNodeBsdfPrincipled")
principled_2.location = (1000.0, 0.0)
displacement = nodes.new(type="ShaderNodeDisplacement")
displacement.location = (1300.0, -200.0)
texture = nodes.new(type="ShaderNodeTexImage")
texture.location = (-200.0, 0.0)
mapping = nodes.new(type="ShaderNodeMapping")
mapping.location = (-400.0, 0.0)
coordinate = nodes.new(type="ShaderNodeTexCoord")
coordinate.location = (-600.0, 0.0)
ramp_1 = nodes.new(type="ShaderNodeValToRGB")
ramp_1.location = (1000.0, -700.0)
ramp_2 = nodes.new(type="ShaderNodeValToRGB")
ramp_2.location = (700.0, 600.0)
textureBase = nodes.new(type="ShaderNodeTexImage")
textureBase.location = (350.0, 100.0)


# With names
link = links.new(principled_1.outputs["BSDF"], mix.inputs[1])  # 'Shader'
link = links.new(principled_2.outputs["BSDF"], mix.inputs[2])  # 'Shader'
link = links.new(mix.outputs["Shader"], output.inputs["Surface"])
link = links.new(displacement.outputs["Displacement"], output.inputs["Displacement"])
link = links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
link = links.new(coordinate.outputs["UV"], mapping.inputs["Vector"])
link = links.new(texture.outputs["Color"], ramp_1.inputs["Fac"])
link = links.new(texture.outputs["Color"], ramp_2.inputs["Fac"])
link = links.new(ramp_1.outputs["Color"], displacement.inputs["Height"])
link = links.new(ramp_2.outputs["Color"], mix.inputs["Fac"])
link = links.new(textureBase.outputs["Color"], principled_1.inputs["Base Color"])
camera_data = bpy.data.cameras.new(name="myCamera")
camera_object = bpy.data.objects.new("myCamera", camera_data)
bpy.context.scene.collection.objects.link(camera_object)
camera_object.location = (0, 0, 50)


#
bpy.context.object.active_material.cycles.displacement_method = "BOTH"

#
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.feature_set = "EXPERIMENTAL"
bpy.context.scene.cycles.device = "GPU"


#
texture.image = bpy.data.images.load("C:/Users/213a/Documents/Blender API/image.png")
textureBase.image = bpy.data.images.load(
    "C:/Users/213a/Documents/Blender API/text1.jpg"
)
displacement.inputs[1].default_value = 0.5  # midlevel
displacement.inputs[2].default_value = 0.25  # scale


plane.data.materials.append(mat)


# Create light datablock
light_data = bpy.data.lights.new(name="my-light-data", type="SUN")
light_data.energy = 7
# Create new object, pass the light data
light_object = bpy.data.objects.new(name="myLight", object_data=light_data)
# Link object to collection in context
bpy.context.collection.objects.link(light_object)
# Change light position
light_object.location = (200, 200, 15)


#
camera_data = bpy.data.cameras.new(name="myCamera")
camera_object = bpy.data.objects.new("myCamera", camera_data)
bpy.context.scene.collection.objects.link(camera_object)
camera_object.location = (0, 0, 50)


#
bpy.context.object.active_material.cycles.displacement_method = "BOTH"

#
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.cycles.feature_set = "EXPERIMENTAL"
bpy.context.scene.cycles.device = "GPU"
