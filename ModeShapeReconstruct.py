from scipy import interpolate
import numpy as np 
# change all the file locations

# parameterises a random airfoil geometry to the mode shapes
# input airfoil: array [[x,y]]
# from trailing edge as [1,y] it goes over the upper surface to the leading edge [0,0]
# back over the lower surface to the trailing edge [1,y] 
# odd-numbered input array where leading edge lays in the middle
def parameterise_airfoil(oldairfoil):
    
    # modes from the Database:
    # Bouhlel, M. A., He, S., and Martins, J. R. R. A., “mSANN Model Benchmarks,” Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
    modes = np.loadtxt("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/modes.subsonic")
    xcors = modes[0,:].copy()
    nte = 15
    xnew = xcors[nte+1:0-nte-1].copy()

    camber_modes = modes[1:8,0:141]
    thickness_modes = modes[8:16,0:141]

    N = xnew.shape[0]
    Nf = int((N-1)/2) + 1
    scinewairfoil = np.zeros((N,2))

    # interpolating the airfoil geometry as a B Spline
    tck, u = interpolate.splprep([oldairfoil[:,0],oldairfoil[:,1]], s=0)
    LEs = 0.51
    for i in range(oldairfoil.shape[0]):
        if oldairfoil[i,0] == 0.0:
            LEs = u[i]
            break
    def findy(sa,sb,xtemp):
        # find the s value in [sa,sb], and make the x value of this s equals xtemp
        # return the y value of this s
        xfind = 0.0
        upper = sb
        lower = sa
        pa = interpolate.splev(sa, tck)
        pb = interpolate.splev(sb, tck)
        xa = pa[0]# curve.getValue(lower)[0]
        xb = pb[0]# curve.getValue(upper)[0]
        # Laplace smoothing algorithm
        eplison = 1.e-10

        if abs(xa - xtemp) < eplison:
           return pa[1],sa
        if abs(xb - xtemp) < eplison:
           return pb[1],sb

        while (abs(xfind - xtemp) > eplison):
            news = (upper + lower)/2.0
            pnew = interpolate.splev(news, tck)
            xnew = pnew[0]
            if abs(xtemp - xnew) < eplison:
               return pnew[1],news

            if (xtemp - xnew)*(xtemp - xa) > 0.0:
               lower = news
            else:
               upper = news
        return  pnew[1],news

    #upper surface
    s1 = 0.0
    startp = interpolate.splev(s1, tck)
    for i in range(Nf-1):
        if xnew[i] > startp[0]:
            # use interpolation
            startp1 = interpolate.splev(0.0001, tck)
            myk = (startp1[1] - startp[1])/(startp1[0] - startp[0])
            scinewairfoil[i,1] = startp[1] - (startp[0] - xnew[i])*myk
            scinewairfoil[i,0] = xnew[i]
        else:
            # find along the curve
            mysa = s1
            mysb = min(s1+0.3,LEs-1.e-8)
            scinewairfoil[i,0] = abs(xnew[i])
            ytemp,s1 = findy(mysa,mysb,xnew[i])
            scinewairfoil[i,1] = ytemp
    #LE, we assume the leading edge is (0,0)
    scinewairfoil[Nf-1,0] = 0.0
    scinewairfoil[Nf-1,1] = 0.0

    #lower surface
    s1 = LEs + 1.e-8
    endp = interpolate.splev(1.0, tck)
    for i in range(Nf,N):
        if xnew[i] > endp[0]:
            # use interpolation
            endp1 = interpolate.splev(0.9999, tck)
            myk = (endp1[1] - endp[1])/(endp1[0] - endp[0])
            scinewairfoil[i,1] = endp[1] - (endp[0] - xnew[i])*myk
            scinewairfoil[i,0] = xnew[i]
        else:
            mysa = s1
            mysb = min(s1+0.3,1.0)
            scinewairfoil[i,0] = abs(xnew[i])
            ytemp,s1 = findy(mysa,mysb,xnew[i])
            scinewairfoil[i,1] = ytemp

    mycamber = np.zeros(Nf)
    mythickr = np.zeros(Nf)
    for i in range(Nf):
        mycamber[i] = (scinewairfoil[i,1] + scinewairfoil[N-i-1,1])*0.5
        mythickr[i] = scinewairfoil[i,1] - scinewairfoil[N-i-1,1]

    # add the 16 points to the thickness line by getting the first value of the "old" thickness line and add to it 
    # an array of length 16 with a linear distribution between the first values and 0
    thickness_LE = mythickr[0]
    thickness_dist = np.linspace(0, thickness_LE, 16)

    mythickrtwo = np.zeros((1,141))
    mythickrtwo[0,:16] = thickness_dist
    mythickrtwo[0,16:141] = mythickr

    #add an array of 16 (0) to the old camber line to get to the same length as the thickness line
    mycambertwo = np.zeros((1,141))
    mycambertwo[0,16:141] = mycamber

    # calculating the camber and thickness mode shapes 
    camber_modeshapes = np.dot(mycambertwo, np.transpose(camber_modes))
    thickness_modeshapes = np.dot(mythickrtwo, np.transpose(thickness_modes))

    # concatenating both mode shapes to one array: thickness mode shapes must be multiplied by two, to work!
    airfoil_modes = np.zeros((1,14))
    airfoil_modes[0,:7] = camber_modeshapes
    airfoil_modes[0,7:] = np.multiply(2, thickness_modeshapes)

    # return the old camber line, thickness line, the interpolated airfoil and the newly obtained airfoil modes
    return mycamber,mythickr, scinewairfoil, airfoil_modes 


def reconstruct_airfoil(airfoil_modes):
    
    # modes from the Database:
    # Bouhlel, M. A., He, S., and Martins, J. R. R. A., “mSANN Model Benchmarks,” Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
    modes = np.loadtxt("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/modes.subsonic")
    # the x-vector for the distribution of the points of the airfoil geometry is saved in the first line of the mode_matrix
    x = modes[0,:].copy()
    mode_matrix = modes[1:,:].copy()

    # computing the y-values of the airfoil using the mode shapes and the mode_matrix
    y_comp = np.dot(airfoil_modes, mode_matrix).flatten()
    

    return x, y_comp


