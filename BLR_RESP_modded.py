"""
BLR-RESP
---------------------------------------------------------------------
C. Martin Gaskell, Peter Harrington
University of California Santa Cruz

"""
from tkinter import *
#from astropy.convolution import convolve, Box1DKernel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = '14.0'

# Basic parameters
screenwidth = 1280
screenheight = 768
npoints = 1500
nellipses = 20

# Set up graphics window 
root = Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
canv = Canvas(root, bg='#000000', bd=0, width=screenwidth, height=screenheight)
canv.grid(column=0, row=0, sticky=(N, W, E, S))

# Simulation model parameters
tilt = 17 # Inclination of the disk (in degrees) to the observer's line 
         # of sight (0 degrees = face-on)
halfopen = 80 # Half-opening angle of BLR (make this = 90 degrees for thin disk)
rflare = 0 # was 95
phiflare = 80  # about 85 degrees seems to be perpendicular; 
               # currently 90 deg is towards the observer, 270 is away
zflare = 0 # height of flare above plane (+ve is away from observer = shorter lag)
rinner = 150 # inner radius of BLR
router = 500 # outer radius of BLR
alpha = 2.0 # power-law index for cloud response (i.e. alpha = 2 gives off-axis 
            # inverse-square law)

# Black hole coords
xBH = screenwidth/2 
yBH = screenheight/2

# Indices to highlight a given velocity
loind = 435
hiind = 465

# Blob location
blen = 300 # major axis (x direction) 250--Medium   100--Small
bht = 300
xblob = -200
yblob = 300

xcenter = xBH+xblob
ycenter = screenheight-yBH-yblob

blen2 = 250 # major axis (x direction) 250--Medium   100--Small
bht2 = 450 # minor axis (y direction)   175--Medium   75--Small
xblob2 = -1000
yblob2 = 2000

xcenter2 = xBH+xblob2
ycenter2 = screenheight-yBH-yblob2

# Transmission coefficient
R_v = 3.1
R_Ha = 2.407
R_Hb = 3.612
E_bv = 0.7

A_Ha = R_Ha*E_bv
A_Hb = R_Hb*E_bv
T_Ha = np.power(10, -A_Ha/2.5)
T_Hb = np.power(10, -A_Hb/2.5)

xflare = -rflare*np.cos(phiflare*np.pi/180)
yflare = -rflare*np.sin(phiflare*np.pi/180)
zpflare = yflare*np.sin(tilt*np.pi/180) + zflare*np.cos(tilt*np.pi/180)
broaden = 0
soften = 30
soften2 = soften*soften
inflow = 0.0 # Fractional inflow velocity - try 0.01 to 0.1
wind = 0.00 # Keep this parameter small
fluff = 0.0 # How much the edges are fluffed up by
velwidth = 30000 # Full width at zero intensity range binned and plotted (units?)
FWHM = 90000 # was 80000
evel = 30000 # to mark in the image where a given velocity comes from.  
             # Make this very large or very small to skip.
ewidth = 500 #  width of the previous velocity
speed = 10*npoints/100

tor_rad = 400 # Radius of torus cross-section (was 300)
clumpsize = 20 # radius of dust clumps within torus
 

# Pre-calculate sine and cosine of tilt
costilt = np.cos(tilt*np.pi/180) 
sintilt = np.sin(tilt*np.pi/180)

# Initialize arrays to zero
pt_IDs = [0] * npoints # point IDs
cosi = [0] * npoints
cosphi = [0] * npoints
emissivity = [0] * npoints
dfr = [0] * npoints
p = [0] * npoints
q = [0] * npoints
r = [0] * npoints
sini = [0] * npoints
sinphi = [0] * npoints
theta = [0] * npoints
visible = [True] * npoints
vrad = [0] * npoints
vtheta = [0] * npoints
x = [0] * npoints
xlast = [0] * npoints
xp = [0] * npoints
y = [0] * npoints
ylast = [0] * npoints
yp = [0] * npoints
z = [0] * npoints
zlast = [0] * npoints
zp = [0] * npoints

profile = [0] * npoints
profile_unobs = [0] * npoints
profile_a = [0]*npoints
profile_b = [0]*npoints
number = [0]*npoints

lag = [0.] * npoints
lag_v = np.zeros(npoints)
lag_vcount = np.zeros(npoints)

transfer = [0.] * npoints

# Create black hole
BH = canv.create_oval(screenwidth/2 - 2.5, screenheight/2 - 2.5, 
                     screenwidth/2 + 5, screenheight/2 + 5, fill='cyan')

# Create flare
FL = canv.create_oval(xBH+xflare, screenheight-yBH+1-yflare*costilt,
                      xBH+xflare + 3, screenheight-yBH+4-yflare*costilt, fill ='green')

dust_brown = '#4f3c32'
# Create elliptical blob
BL = canv.create_oval(xBH+xblob-0.5*blen, screenheight-yBH-yblob-0.5*bht,
                      xBH+xblob+0.5*blen, screenheight-yBH-yblob+0.5*bht,
                      fill = dust_brown)
BL2 = canv.create_oval(xBH+xblob2-0.5*blen2, screenheight-yBH-yblob2-0.5*bht2,
                      xBH+xblob2+0.5*blen2, screenheight-yBH-yblob2+0.5*bht2,
                      fill = dust_brown)
"""

# Create rectangular blob
BL = canv.create_rectangle(xBH+xblob-0.5*blen, screenheight-yBH-yblob-0.5*bht,
                      xBH+xblob+0.5*blen, screenheight-yBH-yblob+0.5*bht,
                      fill ='green')
"""

# Calculate and render cutoff ellipse
v = np.pi - (tilt*(np.pi/180))
R = router + tor_rad + clumpsize
re = R + tor_rad*np.cos(v)   
zshift = tor_rad*np.sin(v)*sintilt
for t in np.arange(0, 2*np.pi, np.pi/100):
    xe = xBH + re*np.cos(t)
    ye = screenheight - yBH - zshift - re*np.cos(tilt*np.pi/180)*np.sin(t)
    canv.create_oval(xe-3, ye-3, xe+3, ye+3, fill='red')
ecenter = yBH + zshift
majax = re
minax = re*costilt

# Draw torus ellipses
for i in range(nellipses):
     v = np.pi - np.arcsin(i/((nellipses-1)*1.0))
     R = router + tor_rad + clumpsize
     re = R + tor_rad*np.cos(v)   
     zshift = tor_rad*np.sin(v)*sintilt
     
     for t in np.arange(0, 2*np.pi, np.pi/100):
         xe = xBH + re*np.cos(t)
         ye = screenheight - yBH - zshift - re*np.cos(tilt*np.pi/180)*np.sin(t)
         #drawing outflow
         #canv.create_oval(xe-2, ye-2, xe+2, ye+2, fill='blue')

# Initialize particle data and plot on screen
for i in range(npoints):
     r[i] = rinner + np.random.rand()*(router-rinner)
     theta[i] = np.random.rand()*2*np.pi # Give the points random phases
     vtheta[i] = speed /(r[i]*np.sqrt(r[i])) # Assign the angular speed according
                                             # to Kepler's 3rd law; change sign to change direction of rotation
     phi = np.random.rand()*np.pi # Gives the rotation of the line of nodes to the 
                               # perpendicular to the line of sight.
     rand1 = np.random.rand()*2-1
     rand2 = np.random.rand() # np.sign(rand1)*np.sqrt(rand2)*
     inc = np.random.rand()*(90-halfopen)*np.pi/180  # Assign inclination of orbit
     sini[i] = np.sin(inc)
     cosi[i] = np.cos(inc)
     sinphi[i] = np.sin(phi)
     cosphi[i] = np.cos(phi)

     theta[i] = theta[i] + vtheta[i]
     #z[i] = np.abs(r[i]*(np.sin(theta[i]*0.71)/2 +0.5)*sini[i]) # Accelerating up and down
     #z[i] = np.abs(r[i]*np.sin(theta[i])*sini[i]) # Explosively ejected off the accretion disc
     z[i] = np.abs(r[i]*np.sin(theta[i])*sini[i]) # Old BLR  
     p[i] = r[i]*np.cos(theta[i])
     q[i] = r[i]*np.sin(theta[i])*cosi[i]
     x[i] = p[i]*cosphi[i] - q[i]*sinphi[i] # these are the physical coordinates in the AGN
     y[i] = p[i]*sinphi[i] + q[i]*cosphi[i]
	 
     # What is plotted is in the observer's coordinate system:
     xp[i] = x[i]
     yp[i] = y[i]*costilt + z[i]*sintilt
     zp[i] = -y[i]*sintilt + z[i]*costilt    
     vrad[i] = -(zp[i] - zlast[i])*FWHM/speed #+ broaden*(Rnd()+Rnd()+Rnd()+Rnd()-2)
     
     # Plot points
     xlast[i] = xp[i]
     ylast[i] = yp[i]
     zlast[i] = zp[i]
     
     pt_x = xp[i]+xBH
     pt_y = screenheight-yp[i]-yBH
     pt_IDs[i] = canv.create_line(pt_x, pt_y, pt_x+1, pt_y+1, fill='white')

steps = 0

# Main loop
def loop(steps):
    for i in range(npoints):
        theta[i] = theta[i] + vtheta[i]
        #z[i] = np.abs(r[i]*(np.sin(theta[i]*0.71)/2 +0.5)*sini[i]) # Accelerating up and down
        #z[i] = np.abs(r(i)*Sin(theta(i))*sini(i)) # Explosively ejected off the accretion disc
        z[i] = np.abs(r[i]*np.sin(theta[i])*sini[i]) # Old BLR  
        p[i] = r[i]*np.cos(theta[i])
        q[i] = r[i]*np.sin(theta[i])*cosi[i]
        x[i] = p[i]*cosphi[i] - q[i]*sinphi[i] # these are the physical coordinates in the AGN
        y[i] = p[i]*sinphi[i] + q[i]*cosphi[i]
	 
        # What is plotted is in the observer's coordinate system:
        xp[i] = x[i]
        yp[i] = y[i]*costilt + z[i]*sintilt
        zp[i] = -y[i]*sintilt + z[i]*costilt    
        vrad[i] = (zp[i] - zlast[i])*FWHM/speed #+ broaden*(np.random.rand()+np.random.rand()+
                                               #             np.random.rand()+np.random.rand() - 2)
        # Make clouds hidden by dust invisible
        #thisx = xp[i]+xBH
        #thisy = yp[i]+yBH
        #visible[i] = check_if_visible(thisx,thisy)
        
        # Add weighted photons from flare    
        vindex = int(vrad[i]*npoints/velwidth+npoints/2)
        dfr[i] = np.sqrt((x[i]-xflare)*(x[i]-xflare) + (y[i]-yflare)*(y[i]-yflare) 
                       + (z[i]-zflare)*(z[i]-zflare) + soften2)
        if (vindex > 1 and vindex < npoints and visible[i]==True):
           if (hide_from_outflow(xp[i]+xBH,screenheight-yp[i]-yBH)==False):
              profile[vindex] += (1/dfr[i])**alpha 
              profile_a[vindex] += 2.72*(1/dfr[i])**alpha    
              profile_b[vindex] += (1/dfr[i])**alpha
           else:
              profile_a[vindex] += 2.72*T_Ha*(1/dfr[i])**alpha    
              profile_b[vindex] += T_Hb*(1/dfr[i])**alpha
           profile_unobs[vindex] += (1/dfr[i])**alpha
           number[vindex] += 1
            
        

        # Lags
        lag[i] = dfr[i] + zp[i]
        if (hide_from_outflow(xp[i]+xBH,screenheight-yp[i]-yBH)==False):
            lag_v[vindex] += lag[i]
            lag_vcount[vindex] += 1
        
        # Transfer Function
        tfindex = int(lag[i]-zpflare)+1
        if (tfindex > 0 and tfindex < 1000 and visible[i]==True):
             if (hide_from_outflow(xp[i]+xBH,screenheight-yp[i]-yBH)==False):
                transfer[tfindex] += 1
        
        # Inflow, wind, and population maintenance
        r[i] = r[i] - r[i]*vtheta[i]*inflow 
        if (r[i] < rinner + (np.random.rand()-0.5)*rinner*fluff):
           r[i] = router +(np.random.rand()-0.5)*router*fluff # replace points and fluff the outer edge
        if (wind > 0): r[i] = r[i]*(1+r[i]*wind/1000)

        
        # Plot points
        prevx = xlast[i]+xBH
        prevy = screenheight-ylast[i]-yBH
        mv_x = xp[i]+xBH - prevx
        mv_y = screenheight-yp[i]-yBH - prevy
        
        '''
        if (dfr[i] < 100):
            canv.itemconfig(pt_IDs[i], fill='green') # color flare-enhanced clouds
        else: canv.itemconfig(pt_IDs[i], fill='white')    
        #if (np.abs(vrad[i]+evel) < ewidth): # "enhance some velocity"
            #canv.move(pt_IDs[i], mv_x, mv_y)
        '''
        canv.itemconfig(pt_IDs[i], fill='white') 
        if vindex >= loind and vindex<= hiind: canv.itemconfig(pt_IDs[i], fill='green')
        
        if visible[i] == False:
            canv.itemconfig(pt_IDs[i], fill='red')
        if (hide_from_outflow(xp[i]+xBH,screenheight-yp[i]-yBH)==True):
            canv.itemconfig(pt_IDs[i], fill='red')
        
        
        canv.move(pt_IDs[i], mv_x, mv_y)
        
        xlast[i] = xp[i]
        ylast[i] = yp[i]
        zlast[i] = zp[i]
        

    # Recursively update simulation every 30 ms
    steps += 1
    root.after(30, loop, steps)

# Checks to see if the point at px, py, should be hidden according to
# ellipses drawn on the surface of the torus. Returns boolean for
# visibility status at given index

def check_if_visible(px, py):
    term1 = ((px-xBH)/(1.0*majax))**2
    term2 = ((py-ecenter)/(1.0*minax))**2
    return term1 + term2 <= 1

# elliptical outflow hiding
def hide_from_outflow(px,py):
    term1 = ((px-xcenter)/(0.5*blen))**2
    term2 = ((py-ycenter)/(0.5*bht))**2
    behind1 = (term1 + term2 <= 1)
    
    term1 = ((px-xcenter2)/(0.5*blen2))**2
    term2 = ((py-ycenter2)/(0.5*bht2))**2
    behind2 = (term1 + term2 <= 1)
    
    # to change to "hole" add not??
    return (behind1 or behind2)
    
"""
# rectangular outflow hiding
def hide_from_outflow(px,py):
    term1 = np.abs(px-xcenter) <= 0.5*blen
    term2 = np.abs(py-ycenter) <= 0.5*bht
    return (term1 and term2)
"""

# End of simulation routine: Plot data and dump to file
def end_sim():
    lineprofile = np.array(profile)
    lineprofile_unobs = np.array(profile_unobs)
    obj = 'upwards'
    maximum = max(lineprofile)
    np.savetxt(obj + "_obsc_prof.txt", lineprofile)
    np.savetxt(obj + "_unobsc_prof.txt", lineprofile_unobs)
    lineprofile[lineprofile==0] = np.nan
    lineprofile_unobs[lineprofile_unobs==0] = np.nan
    indices = []
    for i in range(len(profile)):
        indices.append(i)
    profind = np.subtract(indices,0.5*npoints)
    profind = np.multiply(profind,-25)
    #can also save profind (velocities) as txt
    np.savetxt("velocities.txt", profind)
    
    
    threshold = .95
    
    level = maximum*threshold
    keepx = []
    keepy = []
    for i in range(len(lineprofile)):
        if lineprofile[i] > level:
            keepx.append(profind[i])
            keepy.append(lineprofile[i])
            
    np.savetxt("keepx.txt", keepx)
    np.savetxt("keepy.txt", keepy)
    
    lower_sum = 0
    upper_sum = 0
    
    for i in range(len(keepx)):
        lower_sum = lower_sum + keepy[i]-level
        upper_sum = upper_sum + (keepy[i]-level)*keepx[i]
    
    weighted_mean = upper_sum/lower_sum
    
    print('weighted mean: ' + str(weighted_mean))
    
    
    plt.figure(1)
    plt.plot(profind, lineprofile, 'r-', label="Obscured", linewidth=1.0)
    plt.plot(profind, lineprofile_unobs, 'b-', label="Unobscured", linewidth=1.0)
    #plt.plot([0,0], [0, max(profile)], 'k--')
    plt.autoscale(axis=y)
    #plt.annotate("y="+str(yblob), xy=(-9500,0.9*max(profile)))
    #plt.xlim((-15000,15000))
    plt.tick_params(axis='y', labelleft='off')
    #plt.xlabel("Radial Velocity (km$\cdot$s$^{-1}$)")
    plt.ylabel("Relative Intensity")
    #plt.xticks([-10000,-5000, 0, 5000, 10000], ["-10000", "-5000", "0", "5000", "10000"])
    plt.tight_layout()
    #plt.savefig("TotalExtinctionFig_Prof4.pdf")
    numfig = 7
    plt.savefig('Going_up_' + str(numfig) + '_profile_Grace.eps', format='eps')
    
    prof_a = np.array(profile_a)
    prof_b = np.array(profile_b)
    prof_a[prof_a==0] = np.nan
    prof_b[prof_b==0] = np.nan
    #prof_a = convolve(prof_a, Box1DKernel(10))
    #prof_b = convolve(prof_b, Box1DKernel(10))
    plt.figure(3)
    plt.plot(profind, prof_a/prof_b, 'k-')
    print (np.nansum(prof_a)/np.nansum(prof_b))
    root.destroy()


# Setup function calls to update simulation, plot values, and exit
root.after(30, loop, steps)
root.protocol("WM_DELETE_WINDOW", end_sim)

# Execute
root.mainloop()