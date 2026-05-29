"""Helper functions for implementing the formulas described in https://arxiv.org/abs/2403.13321."""

import jax.numpy as jnp
from jax import Array


def hover_induced_velocity(
    mass: Array,
    gravitational_acceleration: Array,
    air_density: Array,
    propeller_radius: Array,
    number_propellers: int,
) -> Array:
    """This function computes the induced velocity of a drone at hover."""
    return jnp.sqrt(
        mass
        * gravitational_acceleration
        / (2 * jnp.pi * air_density * propeller_radius**2 * number_propellers)
    )



def jet_centerline_velocity(
    hover_induced_velocity: Array,
    s: Array,
    motor_distance: Array,
    Bd: float = 10.11,
    s0: float = -5.817,
) -> Array:
    """This function computes the velocity of the jet at distance r=0 along s."""
    return hover_induced_velocity * Bd / (s / motor_distance - s0)


def jet_half_width(
    motor_distance: Array,
    s: Array,
    spreading_rate: float = 0.07668,
    s0: float = -5.817,
) -> Array:
    """This function computes the cone of the jet where the velocity from the centerline halves."""
    return motor_distance * spreading_rate * (s / motor_distance - s0)


def jet_radial_profile(
    jet_centerline_velocity: Array,
    r: Array,
    jet_half_width: Array,
) -> Array:
    """This function computes the downwash velocity field."""
    return jet_centerline_velocity / (
        1 + (jnp.sqrt(2) - 1) * (r / jet_half_width) ** 2
    ) ** 2
