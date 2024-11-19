# Robot Model Creation Guide

In this project, we chose to use MuJoCo as the simulator and Pinocchio as the framework to extract main functions for control. Typically, the robot is defined by a .urdf file, which is useful for using with the Gazebo simulator within the ROS/ROS2 framework at subsequent steps.

## Converting .urdf to .xml

To convert a .urdf file to a .xml file for using MuJoCo, follow these simple steps:

**Note:** The attribute `fusestatic="false"` is needed to ensure that `base_link` is preserved.

1. Insert the following lines into your .urdf file:

```xml
<mujoco>
    <compiler meshdir="/path_to_meshes" discardvisual="false" fusestatic="false"/>
</mujoco>
```

Here, `/path_to_meshes` is the relative path to the meshes directory, relative to the .xml file.

- Start the GUI of MuJoCo from a shell

    ```bash
    cd /path_to_save_xml
    python3 -m mujoco.viewer
    ```

    drag the .urdf file, then click the appropriate button `Save xml` to save .xml file in current directory `/path_to_save_xml`.
    
    
- Alternatively, you can use the following Python script to convert the .urdf file to .xml:
    
    ```python
    import mujoco
    model = mujoco.MjModel.from_xml_path('*.urdf')
    mujoco.mj_saveLastXML('*.xml', model)
    ```

## Optional: Wrap with a scene.xml

You can add a `scene.xml` file to include the robot, along with a textured groundplane, skybox, and haze.
