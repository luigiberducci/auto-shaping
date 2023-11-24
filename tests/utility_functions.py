from shaping import Variable, Constant


def get_cartpole_spec_within_xlim():
    specs = [
        'ensure abs "x" <= 2.4',
    ]
    constants = []
    variables = [
        Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
        Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
        Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
        Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
    ]
    return specs, constants, variables


def get_cartpole_spec_within_xlim_and_balance():
    specs = [
        'ensure abs "x" <= 2.4',
        'ensure abs "theta" <= 0.2',
    ]
    constants = []
    variables = [
        Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
        Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
        Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
        Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
    ]
    return specs, constants, variables


def get_cartpole_example1_spec():
    specs = [
        'ensure abs "x" <= 2.4',
        'encourage abs "theta" <= 0.2',
        'achieve abs "x" <= 0.25',
    ]
    constants = []
    variables = [
        Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
        Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
        Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
        Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
    ]
    return specs, constants, variables


def get_cartpole_example2_spec():
    specs = [
        'achieve "dist" < 2.4',
        'ensure abs "theta" < 0.2',
        'ensure abs "x" < 2.4',
    ]
    constants = [
        Constant(name="x_goal", value=0.0),
        Constant(name="axle_y", value=100.0),
        Constant(name="pole_length", value=1.0),
        Constant(name="y_goal", value="axle_y + pole_length"),
    ]
    variables = [
        Variable(name="x", fn="state[0]", min=-2.4, max=2.4),
        Variable(name="x_dot", fn="state[1]", min=-3.0, max=3.0),
        Variable(name="theta", fn="state[2]", min=-0.2, max=0.2),
        Variable(name="theta_dot", fn="state[3]", min=-3.0, max=3.0),
        Variable(name="y", min=0.0, max=110.0, fn="axle_y + pole_length*np.cos(theta)"),
        Variable(
            name="dist", min=0.0, max=2.4, fn="np.sqrt((x-x_goal)**2 + (y-y_goal)**2)"
        ),
    ]
    return specs, constants, variables


def get_bipedal_walker_safety():
    specs = ['achieve "const" < 0', 'ensure "collision" <= 0.0']
    constants = [
        Constant(name="dist_hull_limit", value=0.225),
    ]
    variables = [
        Variable(name="const", fn="1.0", min=0.0, max=1.0),
        Variable(
            name="collision", fn="-1.0 if env.game_over else 1.0", min=-1.0, max=1.0
        ),
    ]

    return specs, constants, variables


def get_bipedal_walker_safety_minlidar():
    specs = [
        'achieve "const" < 0',
        'ensure "min_lidar" >= "dist_hull_limit"',
    ]
    constants = [
        Constant(name="dist_hull_limit", value=0.225),
    ]
    variables = [
        Variable(name="const", fn="1.0", min=0.0, max=1.0),
        Variable(name="min_lidar", fn="np.min(state[-10:])", min=0.0, max=1.0),
    ]

    return specs, constants, variables


def get_bipedal_walker_achieve_norm():
    """
    Write achieve w.r.t. a normalized distance "remaining_x".
    """
    specs = [
        'achieve "remaining_x" <= 0.0',
    ]
    constants = [
        Constant(name="x_goal", value=88.667),
    ]
    variables = [
        Variable(
            name="remaining_x",
            fn="(x_goal - env.hull.position[0])/x_goal",
            min=0.0,
            max=1.0,
        ),
    ]

    return specs, constants, variables


def get_bipedal_walker_achieve_unnorm():
    """
    Write achieve x_goal without normalization.
    """
    specs = [
        'achieve "x" >= "x_goal"',
    ]
    constants = [
        Constant(name="x_goal", value=88.667),
    ]
    variables = [
        Variable(name="x", fn="env.hull.position[0]", min=0.0, max=88.667),
    ]

    return specs, constants, variables
