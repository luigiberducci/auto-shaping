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
