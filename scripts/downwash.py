import jax.numpy as jnp
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from downwash_helpers import (
    hover_induced_velocity,
    jet_centerline_velocity,
    jet_half_width,
    jet_radial_profile,
)
from jax.scipy.spatial.transform import Rotation as R
from numpy.typing import NDArray


def downwash_force_fn(data: SimData) -> SimData:
    """Compute the current downwash disturbance for all drone pairs.

    Shape convention used in the pairwise terms:
        (n_worlds, target_drone, target_rotor, source_drone, ...)

    Current modeling assumptions:
        * each source drone produces a vertical world-frame jet below its COM,
        * source drones do not disturb themselves,
        * contributions from multiple source drones are summed at each target rotor,
        * the final disturbance is the difference between drag in downwash air
          and drag in still air.
    """
    states = data.states
    pos = states.pos
    quat = states.quat
    mass = data.params.mass
    arm_length = data.params.L
    prop_radius = data.params.prop_radius
    gravity_vec = data.params.gravity_vec
    mixing_matrix = data.params.mixing_matrix

    rot = R.from_quat(quat).as_matrix()

    # Infer rotor locations from the first two rows of the mixer. For an x-configuration quadrotor
    # these signs encode the roll/pitch moment arms. The z offsets are zero because all rotors lie
    # in the body xy-plane.
    rotor_offsets_body = (arm_length / jnp.sqrt(2)) * jnp.stack(
        (
            -mixing_matrix[..., 1, :],
            mixing_matrix[..., 0, :],
            jnp.zeros_like(mixing_matrix[..., 0, :]),
        ),
        axis=-1,
    )

    rotor_offsets_world = (rot[..., None, :, :] @ rotor_offsets_body[..., :, None])[..., 0]
    rotor_pos_world = pos[..., None, :] + rotor_offsets_world  # (n_worlds, n_drones, n_rotors, 3)

    # Build pairwise target-rotor to source-COM offsets. Positive s means the target rotor is below
    # the source drone, which is the only region where this far-field jet model is applied.
    pos_differences = rotor_pos_world[:, :, :, None, :] - pos[:, None, None, :, :]
    r = jnp.linalg.norm(pos_differences[..., :2], axis=-1)
    s = -pos_differences[..., 2]

    motor_distance_source = 2.0 * arm_length

    # The half-width is the radial distance where the empirical profile reaches half of the
    # centerline velocity. We use it both as the radial profile scale and as a finite cone cutoff.
    cone_border = jet_half_width(motor_distance=motor_distance_source, s=s)

    n_drones = pos.shape[1]
    not_self = ~jnp.eye(n_drones, dtype=bool)[
        None, :, None, :
    ]  # (n_worlds, target_drone, target_rotor, source_drone)

    in_far_field = (s > 0.0) & (r < cone_border) & not_self

    # Hover induced velocity at the rotor disk from momentum theory. This is the paper's reference
    # velocity for scaling the turbulent jet model.
    v_hover = hover_induced_velocity(
        mass=mass,
        gravitational_acceleration=jnp.linalg.norm(gravity_vec),
        air_density=1.225,
        propeller_radius=prop_radius,
        number_propellers=4,
    )
    v_hover_source = v_hover[:, None, None, :, 0]

    # First pass: compute the downwash speed created by each source drone assuming hover in still
    # air. The pairwise values are masked to remove self-interactions and drones outside the cone.
    v_center = jet_centerline_velocity(
        hover_induced_velocity=v_hover_source, s=s, motor_distance=motor_distance_source
    )
    v_down = jet_radial_profile(jet_centerline_velocity=v_center, r=r, jet_half_width=cone_border)
    v_down = jnp.where(in_far_field, v_down, 0.0)

    # Average the rotor-level incoming flow into a per-drone value. This is used to adjust the
    # target drone's induced velocity when it is already ingesting downwash from another drone.
    total_v_down_per_rotor = jnp.sum(v_down, axis=-1)
    total_v_down = jnp.mean(total_v_down_per_rotor, axis=-1, keepdims=True)

    # Momentum-theory correction for nonzero incoming axial flow at the target propellers.
    v_hover_adjusted = total_v_down / 2 + jnp.sqrt((total_v_down / 2) ** 2 + v_hover**2)

    v_hover_source = v_hover_adjusted[:, None, None, :, 0]

    # Second pass: recompute the jet field with the adjusted induced velocity and keep the final
    # incoming downwash speed at each target rotor.
    v_center = jet_centerline_velocity(
        hover_induced_velocity=v_hover_source, s=s, motor_distance=motor_distance_source
    )
    v_down = jet_radial_profile(jet_centerline_velocity=v_center, r=r, jet_half_width=cone_border)
    v_down = jnp.where(in_far_field, v_down, 0.0)
    total_v_down_per_rotor = jnp.sum(v_down, axis=-1)

    # Convert scalar downwash speeds into world-frame wind vectors at the target rotor locations.
    # Negative z means air moving downward in the world frame.
    wind_world_rotor = jnp.zeros_like(rotor_pos_world)
    wind_world_rotor = wind_world_rotor.at[..., 2].set(-total_v_down_per_rotor)

    target_z_world = rot[..., :, 2]

    # Rotor-local relative air velocity. The disturbance is modeled through the change in
    # aerodynamic drag compared with the same vehicle state in still air.
    rel_air_world_rotor = states.vel[..., None, :] - wind_world_rotor
    v_a_body_rotor = (rot.mT[..., None, :, :] @ rel_air_world_rotor[..., None])[..., 0]
    v_a_body_still = (rot.mT @ states.vel[..., None])[..., 0]

    # Parasitic drag at the COM. The rotor-level wind is projected onto the target body z-axis and
    # averaged into one equivalent axial inflow per drone.
    total_v_down_com = -jnp.mean(
        jnp.sum(wind_world_rotor * target_z_world[..., None, :], axis=-1), axis=-1
    )

    wind_world = jnp.zeros_like(pos)
    wind_world = wind_world.at[..., 2].set(-total_v_down_com)
    rel_air_world = states.vel - wind_world
    v_a_body = (rot.mT @ rel_air_world[..., None])[..., 0]
    speed = jnp.linalg.norm(v_a_body, axis=-1, keepdims=True)
    speed_still = jnp.linalg.norm(v_a_body_still, axis=-1, keepdims=True)

    C_diag = jnp.array([-2.329916287671239e-05, -2.329916287671239e-05, -3.078507303977562e-05])
    parasitic_drag = speed * C_diag * v_a_body
    parasitic_drag_still = speed_still * C_diag * v_a_body_still

    # Rotor drag at each rotor includes the body translational air velocity and the local velocity
    # from body angular rate. The moment contribution is r x F around the vehicle COM.
    K_diag = jnp.array([-2.1991768793537817e-07, -2.1991768793537817e-07, -1.7024656365051572e-07])
    rotor_vels_body = v_a_body_rotor + jnp.cross(states.ang_vel[..., None, :], rotor_offsets_body)
    rotor_vels_body_still = v_a_body_still[..., None, :] + jnp.cross(
        states.ang_vel[..., None, :], rotor_offsets_body
    )

    rotor_drag = K_diag * states.rotor_vel[..., None] * rotor_vels_body
    rotor_drag_still = K_diag * states.rotor_vel[..., None] * rotor_vels_body_still
    rotor_drag_summed = jnp.sum(rotor_drag - rotor_drag_still, axis=-2)

    torque_body = jnp.sum(jnp.cross(rotor_offsets_body, rotor_drag - rotor_drag_still), axis=-2)
    torque_world = (rot @ torque_body[..., None])[..., 0]

    force_body = parasitic_drag - parasitic_drag_still + rotor_drag_summed
    force_world = (rot @ force_body[..., None])[..., 0]

    data = data.replace(states=states.replace(force=force_world, torque=torque_world))

    return data
