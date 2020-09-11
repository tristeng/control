# Control Experiments
This repository outlines various control experiments using modern control theory and state space controller design.

The results are simulated using [pymunk](https://www.pymunk.org) (a 2D physics simulator) and 
[pyglet](http://www.pyglet.org)

## Requirements
Modern version of Python (developed on Python 3.7.1).

It is highly recommended that you create a Python virtual environment before installing the python libraries.

### To run the simulations in a game engine:  
```pip install pymunk pyglet```

### To run the notebooks:  
```pip install numpy scipy matplotlib jupyter sympy```

## Running the Samples

### Simulations
#### Inverted Pendulum on a Moveable Cart
Controlling an inverted pendulum using a moveable cart. The simulation commands the cart to various positions every
few seconds while still keeping the pendulum stable in a vertical position.

```python invpend.py```

#### Double Inverted Pendulum on a Moveable Cart
Controlling a double inverted pendulum using a moveable cart. The simulation commands the cart to various positions every
few seconds while still keeping both pendulums stable in a vertical position.

```python dinvpend.py```

#### Quadcopter in 2D space
Controlling a quadcopter in 2D space. The simulation commands the quadcopter to several waypoints every few seconds.

```python quadcopter2d.py```

### Notebooks
Start Jupyter:  
```jupyter notebook``` and go to http://localhost:8888 and load the notebook you desire

The notebooks derive the motions of equation using sympy (symbolic math library) and then design controllers. 
The results from the notebooks are used to test the results in pymunk and pyglet.
