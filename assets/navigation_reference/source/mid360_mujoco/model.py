"""Generate MJCF from the existing robot's physical configuration."""
from pathlib import Path
import math
import json
import xml.etree.ElementTree as E
import yaml
from .config import ROOT


def add(parent, tag, **attrs):
    return E.SubElement(parent,tag,{k:str(v) for k,v in attrs.items()})


def vec(x):
    return " ".join(f"{v:.12g}" for v in x)


def build(config, robot, field=None, output=None):
    root=E.Element("mujoco",model="dual_mid360_sentry")
    add(root,"compiler",angle="radian",autolimits="true",inertiafromgeom="auto")
    add(root,"option",timestep=config["physics"]["timestep"],gravity=f"0 0 -{config['physics']['gravity']}",
        integrator="implicitfast",iterations="50",cone="elliptic")
    default=add(root,"default")
    add(default,"geom",friction="0.02 0.001 0.0001",solref="0.008 1",solimp="0.95 0.99 0.001",condim="3")
    visual=add(root,"visual");add(visual,"global",offwidth="960",offheight="540")
    add(visual,"headlight",ambient="0.5 0.5 0.5",diffuse="0.6 0.6 0.6")
    asset=add(root,"asset");world=add(root,"worldbody")
    add(world,"light",pos="0 0 12",dir="-0.3 -0.2 -1",directional="true")
    if field:
        field=Path(field).resolve()
        for i,p in enumerate(sorted((field/"visual").glob("*.stl"))):
            name=f"field_visual_{i}"
            add(asset,"mesh",name=name,file=str(p),inertia="shell")
            add(world,"geom",name=f"field_scan_{i}",type="mesh",mesh=name,group="1",contype="0",conaffinity="0",rgba="0.58 0.58 0.58 1")
        contact_manifest=json.loads((field/"manifest.json").read_text())
        centers={x["file"]:x.get("position",[0,0,0]) for x in contact_manifest["pieces"]}
        for i,p in enumerate(sorted((field/"collision").glob("*.obj"))):
            name=f"field_collision_{i}"
            add(asset,"mesh",name=name,file=str(p),inertia="convex")
            add(world,"geom",name=name,type="mesh",mesh=name,pos=vec(centers[p.name]),group="2",rgba="0.5 0.5 0.5 0",contype="1",conaffinity="1")
        spawn=robot["spawn"]
    else:
        add(world,"geom",name="ground",type="plane",size="15 15 0.1",rgba="0.45 0.48 0.5 1")
        for name,pos,size in [("wall_x",[7,0,1],[.1,8,1]),("wall_y",[0,7,1],[8,.1,1]),
                               ("back_wall",[-7,0,1],[.1,8,1]),("side_wall",[0,-7,1],[8,.1,1])]:
            add(world,"geom",name=name,type="box",pos=vec(pos),size=vec(size),rgba="0.7 0.73 0.75 1")
        add(world,"geom",name="ramp",type="box",pos="3 -3 0.2",size="1 0.8 0.08",euler="0 -0.2 0",rgba="0.5 0.6 0.7 1")
        spawn=[0,0,.03,0]
    c=robot["chassis"];m=robot["mount"]
    body=add(world,"body",name="chassis",pos=vec(spawn[:3]),euler=vec([0,0,spawn[3]]))
    add(body,"freejoint",name="base_free")
    mass=c["mass"];l,w,h=c["length"],c["width"],c["height"]
    add(body,"inertial",pos=vec([0,0,c["center_of_mass_z"]]),mass=mass,
        diaginertia=vec([mass*(w*w+h*h)/12,mass*(l*l+h*h)/12,mass*(l*l+w*w)/12]))
    add(body,"geom",name="chassis_body",type="box",pos=vec([0,0,c["center_z"]]),size=vec([l/2,w/2,h/2]),rgba="0.12 0.18 0.22 1")
    add(body,"geom",name="heading",type="box",pos=vec([.12,0,c["center_z"]+h/2+.004]),size="0.065 0.0125 0.003",contype="0",conaffinity="0",rgba="0.95 0.55 0.1 1")
    for i,(sx,sy) in enumerate(((1,1),(1,-1),(-1,1),(-1,-1))):
        carrier=add(body,"body",name=f"wheel_carrier_{i}",pos=vec([sx*c["wheel_half_length"],sy*c["wheel_half_width"],c["wheel_radius"]]))
        add(carrier,"joint",name=f"wheel_suspension_{i}_joint",type="slide",axis="0 0 1",
            range=vec([-c["suspension_travel_down"],c["suspension_travel_up"]]),
            stiffness=c["suspension_stiffness"],damping=c["suspension_damping"],
            springref=-(mass+.73)*config["physics"]["gravity"]/(4*c["suspension_stiffness"]))
        add(carrier,"inertial",pos="0 0 0",mass="0.05",diaginertia="0.00001 0.00001 0.00001")
        wheel=add(carrier,"body",name=f"mecanum_wheel_{i}_link")
        add(wheel,"joint",name=f"mecanum_wheel_{i}_joint",axis="0 1 0",damping="0.001")
        wr,ww,wm=c["wheel_radius"],c["wheel_width"],c["wheel_mass"]
        add(wheel,"inertial",pos="0 0 0",mass=wm,diaginertia=vec([wm*(3*wr*wr+ww*ww)/12,wm*wr*wr/2,wm*(3*wr*wr+ww*ww)/12]))
        add(wheel,"geom",type="cylinder",size=vec([wr,ww/2]),euler=vec([math.pi/2,0,0]),
            contype="0",conaffinity="0",rgba="0.12 0.12 0.14 1")
        # Same width-bounded rounded tread approximation as the old simulator.
        for j in range(24):
            a=2*math.pi*j/24;r=wr-ww/2
            add(wheel,"geom",name=f"wheel_contact_{i}_{j}",type="sphere",size=ww/2,
                pos=vec([r*math.cos(a),0,r*math.sin(a)]),group="2",rgba="0 0 0 0")
    mount=add(body,"body",name="lidar_mount",pos=vec([m["front_x"],0,0]))
    add(mount,"joint",name="lidar_yaw_joint",axis="0 0 1",damping="0.01")
    add(mount,"inertial",pos="0 0 0.17",mass="0.2",diaginertia="0.002 0.002 0.002")
    add(mount,"geom",type="cylinder",pos="0 0 0.16",size="0.012 0.04",rgba="0.5 0.5 0.5 1")
    for i,sign in enumerate((1,-1)):
        frame=("front_mid360","right_mid360")[i]
        sensor=add(mount,"body",name=frame,pos=vec([0,sign*m["half_spacing"],m["optical_height"]]),
                   euler=vec([-sign*math.radians(m["outward_tilt_deg"]),0,0]))
        add(sensor,"inertial",pos="0 0 -0.02",mass="0.265",diaginertia="0.00015 0.00015 0.00015")
        add(sensor,"geom",name=frame+"_body",type="box",pos="0 0 -0.022",size="0.0325 0.0325 0.022",rgba="0.95 0.57 0.16 1" if i==0 else "0.15 0.65 0.85 1")
        add(sensor,"site",name=frame,size="0.003",rgba="1 0 0 1")
        add(sensor,"site",name=frame+"_imu",pos=vec(config["imu"]["position_in_lidar"]),size="0.002")
        add(mount,"geom",type="capsule",fromto=vec([0,0,.19,0,sign*m["half_spacing"],m["optical_height"]-.02]),size="0.009",rgba="0.5 0.5 0.5 1")
    # Camera looks along robot +X, camera up = robot +Z.
    add(body,"camera",name="fpv",pos=vec(robot["fpv"]["position"]),xyaxes="0 -1 0 0 0 1",fovy="65")
    sensors=add(root,"sensor")
    for frame in ("front_mid360","right_mid360"):
        add(sensors,"accelerometer",name=frame+"_accel",site=frame+"_imu")
        add(sensors,"gyro",name=frame+"_gyro",site=frame+"_imu")
    act=add(root,"actuator")
    add(act,"velocity",name="mount_velocity",joint="lidar_yaw_joint",kv="1",forcerange="-5 5")
    E.indent(root)
    text=E.tostring(root,encoding="unicode")
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(text)
    return text


def robot_config(path=None):
    path=Path(path) if path else ROOT/"configs/robot.yaml"
    return yaml.safe_load(path.read_text())

