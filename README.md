### Installation
To install the environment, clone the repository and install the dependencies:

```bash
./setup_env.sh
```

To run the train:
```bash
python3 scripts/train_multi_ppo.py
```

To run the simulation with a trained model:

```bash
python scripts/record_video.py --model models/ppo_street_best.zip --render human
```

# Kinematics

The *Kinematic Bicycle Model* are represented in the `~otw_env.core.vehicle.kinematics.Vehicle` class.

$$
\dot{x}=v\cos(\psi+\beta) \\ \dot{y}=v\sin(\psi+\beta) \\ \dot{v}=a \\ \dot{\psi}=\frac{v}{l}\sin\beta \\ \beta=\tan^{-1}(1/2\tan\delta), \\
$$

where

- $(x, y)$ is the vehicle position;
- $v$ its forward speed;
- $\psi$ its heading;
- $a$ is the acceleration command;
- $\beta$ is the slip angle at the center of gravity;
- $\delta$ is the front wheel angle used as a steering command.

These calculations appear in the `~otw_env.core.vehicle.kinematics.Vehicle.step` method.

## Longitudinal controller

The longitudinal controller is a simple proportional controller:

$$
a = K_p(v_r - v),
$$

where

- $a$ is the vehicle acceleration (throttle);
- $v$ is the vehicle velocity;
- $v_r$ is the reference velocity;
- $K_p$ is the controller proportional gain, implemented as `~otw_env.core.vehicle.controller.ControlledVehicle.KP_A`.

It is implemented in the `~otw_env.core.vehicle.controller.ControlledVehicle.speed_control` method.

## Lateral controller

The lateral controller is a simple proportional-derivative controller, combined with some non-linearities that invert those of the {ref}`kinematics model <vehicle_kinematics>`.

### Position control

$$
v_{\text{lat},r} = -K_{p,\text{lat}} \Delta_{\text{lat}}, \\ \Delta \psi_{r} = \arcsin \left(\frac{v_{\text{lat},r}}{v}\right),
$$

### Heading control

$$
\psi_r = \psi_L + \Delta \psi_{r}, \\ \dot{\psi}_r = K_{p,\psi} (\psi_r - \psi), \\ \delta = \arcsin \left(\frac{1}{2} \frac{l}{v} \dot{\psi}_r\right), \\
$$

where

- $\Delta_{\text{lat}}$ is the lateral position of the vehicle with respect to the lane center-line;
- $v_{\text{lat},r}$ is the lateral velocity command;
- $\Delta \psi_{r}$ is a heading variation to apply the lateral velocity command;
- $\psi_L$ is the lane heading (at some lookahead position to anticipate turns);
- $\psi_r$ is the target heading to follow the lane heading and position;
- $\dot{\psi}_r$ is the yaw rate command;
- $\delta$ is the front wheels angle control;
- $K_{p,\text{lat}}$ and $K_{p,\psi}$ are the position and heading control gains.

Other simulated vehicles follow simple and realistic behaviors that dictate how they accelerate and
steer on the road. They are implemented in the `~otw_env.core.vehicle.behavior_controller.IDMVehicle` class.

## Longitudinal Behavior

The acceleration of the vehicle is given by the *Intelligent Driver Model* (IDM) 

$$
\dot{v} = a\left[1-\left(\frac{v}{v_0}\right)^\delta - \left(\frac{d^*}{d}\right)^2\right] \\ d^* = d_0 + Tv + \frac{v\Delta v}{2\sqrt{ab}} \\
$$

where $v$ is the vehicle velocity, $d$ is the distance to its front vehicle.
The dynamics are parametrised by:

- $v_0$ the desired velocity, as `IDMVehicle.target_velocity`
- $T$ the desired time gap, as `IDMVehicle.TIME_WANTED`
- $d_0$ the jam distance, as `IDMVehicle.DISTANCE_WANTED`
- $a,\,b$ the maximum acceleration and deceleration, as `IDMVehicle.COMFORT_ACC_MAX` and `IDMVehicle.COMFORT_ACC_MIN`
- $\delta$ the velocity exponent, as `IDMVehicle.DELTA`

It is implemented in `IDMVehicle.acceleration` method.

## Lateral Behavior

The discrete lane change decisions are given by the *Minimizing Overall Braking Induced by Lane change* (MOBIL) model from {cite}`Kesting2007`.
According to this model, a vehicle decides to change lane when:

- it is **safe** (do not cut-in):

$$
\tilde{a}_n \geq - b_\text{safe};
$$

- there is an **incentive** (for the ego-vehicle and possibly its followers):

$$
\underbrace{\tilde{a}_c - a_c}_{\text{ego-vehicle}} + p\left(\underbrace{\tilde{a}_n - a_n}_{\text{new follower}} + \underbrace{\tilde{a}_o - a_o}_{\text{old follower}}\right) \geq \Delta a_\text{th},
$$

where

- $c$ is the center (ego-) vehicle, $o$ is its old follower *before* the lane change, and $n$ is its new follower *after* the lane change
- $a, \tilde{a}$ are the acceleration of the vehicles *before* and *after* the lane change, respectively.
- $p$ is a politeness coefficient, implemented as {py:attr}`IDMVehicle.POLITENESS`
- $\Delta a_\text{th}$ the acceleration gain required to trigger a lane change, implemented as {py:attr}`IDMVehicle.LANE_CHANGE_MIN_ACC_GAIN`
- $b_\text{safe}$ the maximum braking imposed to a vehicle during a cut-in, implemented as {py:attr}`IDMVehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED`

It is implemented in the `IDMVehicle.mobil` method.

```{note}
In the `~otw_env.core.vehicle.behavior_controller.LinearVehicle` class, the longitudinal and lateral behaviours
are approximated as linear weightings of several features, such as the distance and speed difference to the leading
vehicle.
```