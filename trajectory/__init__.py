import numpy as np
import sympy as sp


class MininumTrajectory:
    def __init__(self, numcoeffs: int):
        self.numcoeffs = numcoeffs

        t = sp.symbols('t')
        pos = t**0
        for ii in range(1, numcoeffs):
            pos += t**ii

        vel = pos.diff(t)
        acc = vel.diff(t)
        jerk = acc.diff(t)
        snap = jerk.diff(t)
        crac = snap.diff(t)
        pop = crac.diff(t)

        self.eqs = [pos, vel, acc, jerk, snap, crac, pop]
        self.times = None
        self.polys = None
        self.dims = None

    def coeffs_for_time(self, equations, time):
        retval = np.zeros((len(equations), self.numcoeffs))

        for ii, eq in enumerate(equations):
            # pull out the coefficients - ordered from highest to lowest poly order, e.g.: t^3, t^2, t^1, t^0
            coeffs = eq.as_poly().all_coeffs()

            # apply time
            exp = len(coeffs) - 1
            for idx in range(len(coeffs)):
                coeffs[idx] *= time ** exp
                exp -= 1

            # pad out with zeros if some of the coefficients are 0
            while len(coeffs) < self.numcoeffs:
                coeffs.append(0)

            # reverse them to match up with our coeffs vector
            retval[ii] = list(reversed(coeffs))
        return retval

    def generate(self, points, times):
        if len(points) != len(times) or len(points) < 2:
            raise ValueError('Points and times must be lists of equal length greater than 2')

        for idx, starttime in enumerate(times[:-1]):
            endtime = times[idx+1]
            if endtime <= starttime:
                raise ValueError('Times must be ordered from smallest to largest and cannot overlap')

        self.times = np.array(times)

        n = self.numcoeffs * (len(points) - 1)
        A = np.zeros((n, n))

        # use the first point to figure out how many b vectors we need
        self.dims = len(points[0])
        b = [np.zeros((n, 1)) for _ in range(self.dims)]

        # fill in 3 equations for first segment - velocity, acceleration and jerk are all equal to 0 at start time
        nextrow = 0
        numeqs = 3  # TODO - determine if we can calculate this
        equations = self.eqs[1:1+numeqs]
        A[nextrow:nextrow + numeqs, 0:self.numcoeffs] = self.coeffs_for_time(equations, times[0])
        nextrow += numeqs

        # fill in 3 equations for last segment - velocity, acceleration and jerk are all equal to 0 at end time
        A[nextrow:nextrow + numeqs, n - self.numcoeffs:n] = self.coeffs_for_time(equations, times[-1])
        nextrow += numeqs

        # for all segments...
        for idx, startp in enumerate(points[0:-1]):
            endp = points[idx + 1]
            startt = times[idx]
            endt = times[idx + 1]

            # fill in 2 equations for start and end point passing through the poly
            # start point
            col = idx * self.numcoeffs

            A[nextrow:nextrow + 1, col:col + self.numcoeffs] = self.coeffs_for_time([self.eqs[0]], startt)
            for ii in range(len(points[0])):
                b[ii][nextrow] = startp[ii]
            nextrow += 1

            # end point
            A[nextrow:nextrow + 1, col:col + self.numcoeffs] = self.coeffs_for_time([self.eqs[0]], endt)
            for ii in range(len(points[0])):
                b[ii][nextrow] = endp[ii]

            nextrow += 1

        # for all segments, except last...
        for idx in range(len(points) - 2):
            endt = times[idx + 1]

            # fill in 6 equations for velocity, acceleration, jerk, snap, crackle and pop to ensure they are the same
            # through the transition point evaluate both poly's at the end point
            numeqs = 6  # TODO - determine if we can calculate this
            equations = self.eqs[1:1 + numeqs]
            col = idx * self.numcoeffs
            A[nextrow:nextrow + numeqs, col:col + self.numcoeffs] = self.coeffs_for_time(equations, endt)
            col += self.numcoeffs

            # negate endt coefficients since we move everything to the lhs
            A[nextrow:nextrow + numeqs, col:col + self.numcoeffs] = -self.coeffs_for_time(equations, endt)
            nextrow += numeqs

        # solve the system
        x = [np.linalg.solve(A, b[ii]) for ii in range(len(b))]

        # polys will have rows corresponding to segments, columns corresponding to the point dimensions, and 3rd dim
        # will contain the pos, vel, and acc polys
        self.polys = []
        t = sp.symbols('t')
        for ii in range(len(points)-1):
            col = []
            offset = ii * self.numcoeffs
            for jj in range(len(x)):
                layer = [sp.Poly(reversed(x[jj][offset:offset+self.numcoeffs].transpose()[0]), t)]

                # append on time derivatives up to acceleration
                for kk in range(1, 3):
                    # take the time derivative of the previous poly
                    layer.append(layer[kk-1].diff(t))
                col.append(layer)
            self.polys.append(col)

    def getvalues(self, time):
        """
        Returns an array where rows corresponds to the number of dimensions in points, and columns corresponds to the
        position, velocity, and acceleration for the given time

        :param time: the time to evaluate the polys at
        :return: nump array
        """
        if self.times is None or self.polys is None:
            raise AssertionError('Please generate the trajectory first')

        # find the correct poly index
        if time < self.times[0]:
            time = self.times[0]

        if time > self.times[-1]:
            time = self.times[-1]

        idx = np.argwhere(self.times <= time)[-1][0]

        # the last index time actually means we should use the previous poly
        if idx >= len(self.polys):
            idx = len(self.polys) - 1

        retval = np.zeros((self.dims, 3))
        for jj in range(retval.shape[0]):
            for kk in range(retval.shape[1]):
                retval[jj][kk] = self.polys[idx][jj][kk].eval(time)

        return retval


class MinimumSnapTrajectory(MininumTrajectory):
    def __init__(self):
        super().__init__(8)
