#!/usr/bin/env python3

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def read_apcemm_dir(dir_path,times_minutes=None,var_list=None,var_list_1D=None,verbose=False):
    file_template = 'ts_aerosol_case0_{:02d}{:02d}.nc'
    f_list = []
    t_vec = []
    
    if times_minutes is None:
        # Test every minute
        times_minutes = range(0,60*24 + 1,1)
    
    test_list = [file_template.format(int(np.floor(x/60)),int(np.mod(x,60))) for x in times_minutes]
    output_dir = os.path.join(dir_path,'APCEMM_out')
    output_list = [x for x in os.listdir(output_dir) if x.endswith('.nc')]
    for t, f in zip(times_minutes,test_list):
        # Faster to read the directory contents and test against that
        # than to test each file for existence
        #if os.path.isfile(f_long):
        if f in output_list:
            f_long = os.path.join(dir_path,'APCEMM_out',f)
            f_list.append(f_long)
            t_vec.append(t)
    
    #assert len(t_vec) > 0, f'No APCEMM output found for directory {dir_path}'
    if len(t_vec) == 0:
        return None
    
    if var_list is None:
        var_list = ['Ice aerosol particle number','Ice aerosol volume','Temperature','Effective radius','RHi','H2O','IWC','Pressure']
    if var_list_1D is None:
        var_list_1D = ['Ice Mass']
    # This variable is used to ensure we only check for the variable's presence once
    first_read = True
    data_dict = {'t': t_vec}
    var_list_extended = ['x','y'] + var_list
    for var in var_list_extended + var_list_1D:
        data_dict[var] = []
    for f in f_list:
        with xr.open_dataset(f,decode_times=False) as ds:
            if first_read and verbose:
                print(list(ds.variables))
            for var in var_list_extended:
                data_mini = ds[var][...].values
                data_dict[var].append(data_mini)
            for var in var_list_1D:
                data_dict[var].append(ds[var][0].values)
            first_read = False
    return data_dict

def plot_data(var,data_dict,n_times=None,n_x=4,clim_factor=1,clim_force=None,show_cbar=True,ax_arr=None,ice_contour=False):
    if n_times is None:
        n_times = len(data_dict['x'])
    # Is the user providing the array of axes?
    external_ax = ax_arr is not None
    if external_ax:
        (n_y,n_x) = ax_arr.shape
        f = ax_arr.ravel()[0].figure
    else:
        if n_x is None:
            n_x = n_times
        n_y = int(np.ceil(n_times/n_x))
        f, ax_arr = plt.subplots(n_y,n_x,squeeze=False,figsize=(12,4*n_y))
    xlim = [0,0]
    ylim = [0,0]
    clim = [np.inf,-np.inf]
    im_vec = []
    t_vec = data_dict['t']
    for i_time in range(n_times):
        i_y = int(np.floor(i_time/n_x))
        i_x = int(np.mod(i_time,n_x))
        ax = ax_arr[i_y,i_x]
        time = t_vec[i_time]
        if var == 'H2O_delta':
            h2o = data_dict['H2O'][i_time].copy()
            var_data = h2o - np.tile(h2o[:,0],(h2o.shape[1],1)).transpose()
        else:
            var_data = data_dict[var][i_time]
        im = ax.pcolormesh(data_dict['x'][i_time],data_dict['y'][i_time],var_data)
        #plt.colorbar(im)
        ax.set_title(f'{var}\n{time:d} minutes')
        for lim, lim_ax in zip([xlim, ylim, clim],[ax.get_xlim(),ax.get_ylim(),im.get_clim()]):
            lim[0] = min(lim_ax[0],lim[0])
            lim[1] = max(lim_ax[1],lim[1])
        im_vec.append(im)
        if i_x > 0:
            ax.set_yticklabels([])
        min_val = np.nanmin(var_data)
        #print(f' --> Minimum at {time:03d}m: {min_val:15.4e}')
        
        if ice_contour:
            ax.contour(data_dict['x'][i_time],data_dict['y'][i_time],data_dict['Ice aerosol particle number'][i_time],cmap='plasma')
    if show_cbar:
        cax = f.add_axes([0,0,1,0.03])
        plt.colorbar(im,ax=ax,cax=cax,orientation='horizontal')
    for ax in ax_arr.ravel():
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    # Stretch the color limits (show more of the lower bound)
    if clim_force is None:
        clim_stretch = np.array(clim)
        clim_mid = np.mean(clim)
        clim_width = clim[1] - clim[0]
        clim_stretch[1] = clim[0] + clim_width/clim_factor
        clim_stretch[0] = clim[0]
    else:
        clim_stretch = clim_force
    for im in im_vec:
        im.set_clim(clim_stretch)
    if not external_ax:
        f.tight_layout()
    return im_vec

def plot_multi_data(var,dict_of_dicts,**kwargs):
    # Plot a single variable (2D) for every member of a dictionary
    # Each member of the dictionary is the output from read_apcemm_dir
    n_plots = len(dict_of_dicts)
    im_vec = []
    if 'n_times' in kwargs and kwargs['n_times'] is not None:
        n_x = kwargs['n_times']
    else:
        n_x = len(next(iter(dict_of_dicts.values()))[var])
    f, ax_arr = plt.subplots(n_plots,n_x,figsize=(3*n_x,4*n_plots),squeeze=False)
    i_data = 0
    for sim_name, data_dict in dict_of_dicts.items():
        ax_vec = np.full((1,n_x),None)
        ax_vec[:] = ax_arr[i_data,:]
        im_new = plot_data(var,data_dict,show_cbar=False,ax_arr=ax_vec,**kwargs)
        im_vec += im_new
        if i_data > 0:
            for ax in ax_vec.ravel():
                ax.set_title('')
        # Add text to first plot to indicate which source
        ax = ax_vec.ravel()[0]
        ax.text(0.9, 0.1, f'{sim_name:s}\n{var:s}', horizontalalignment='right',
             verticalalignment='center', transform=ax.transAxes)
        i_data += 1
    # Make sure all axes use the same limits
    lim = [np.inf,-np.inf]
    for im in im_vec:
        lim_ax = im.get_clim()
        lim[0] = min(lim_ax[0],lim[0])
        lim[1] = max(lim_ax[1],lim[1])
    for im in im_vec:
        im.set_clim(lim)
    return im_vec

def interp_apcemm(x_from,y_from,data,x_to,y_to):
    from scipy.interpolate import RegularGridInterpolator
    # DO NOT MESS WITH THIS - the indexing is a nightmare
    interp = RegularGridInterpolator((y_from,x_from),data,bounds_error=False)
    xg,yg = np.meshgrid(x_to,y_to)
    return interp((yg,xg))
    
def plot_data_diff(var,baseline,changed,n_times=None,clim=None):
    # Show difference between the baseline and changed output
    # Each should be a dictionary from "read_apcemm_dir"
    # Output will be an array of 2xT plots:
    # Top row:    variable in the baseline dataset
    # Bottom row: difference resulting from the change to the new dataset
    im_vec = []
    if n_times is not None:
        n_x = n_times
    else:
        n_x = len(baseline[var])
    f, ax_arr = plt.subplots(2,n_x,figsize=(3*n_x,4*2))
    # First plot the baseline, then plot the delta
    ax_vec = np.full((1,n_x),None)
    ax_vec[:] = ax_arr[0,:]
    im_new = plot_data(var,baseline,show_cbar=False,ax_arr=ax_vec,n_times=n_x)
    
    # Do some interpolation
    delta_dict = {}
    delta_dict['t'] = baseline['t']
    delta_dict['x'] = []
    delta_dict['y'] = []
    delta_dict[var] = []
    i = 0
    x_to = np.arange(-1000,5000,10)
    y_to = np.arange(-1000,200,10)
    for base_data, changed_data in zip(baseline[var],changed[var]):
        delta_dict['x'].append(x_to)
        delta_dict['y'].append(y_to)
        base_interp = interp_apcemm(baseline['x'][i],baseline['y'][i],base_data,x_to,y_to)
        changed_interp = interp_apcemm(changed['x'][i],changed['y'][i],changed_data,x_to,y_to)
        delta_dict[var].append(changed_interp - base_interp)
        i += 1
    ax_vec[:] = ax_arr[1,:]
    im_diff = plot_data(var,delta_dict,show_cbar=True,ax_arr=ax_vec,n_times=n_x)
    ## Make sure all axes use the same limits
    lim = 0.0
    for im in im_diff:
        lim_ax = np.nanmax(np.abs(im.get_clim()))
        lim = max(lim_ax,lim)
    for im in im_diff:
        im.set_clim([-lim,lim])
        im.set_cmap('RdBu_r')
    # Match diff limits to raw data
    xlim = ax_arr[0,0].get_xlim()
    ylim = ax_arr[0,0].get_ylim()
    for ax in ax_arr.ravel():
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return f, ax_arr, im_diff

def compare_contours(data_dict,i_time):
    # Show contours of H2O and ice particle number side by side for the same time step, for a single APCEMM simulation
    f,ax_arr = plt.subplots(1,2)
    h2o = data_dict['H2O'][i_time].copy()
    # Subtract the boundary condition
    h2o_bc = h2o[:,-2].copy()
    for i_x in range(h2o.shape[1]):
        h2o[:,i_x] -= h2o_bc
    ice = data_dict['Ice aerosol particle number'][i_time]
    
    ax = ax_arr[0]
    im = ax.contour(data_dict['x'][i_time],data_dict['y'][i_time],ice,cmap='plasma')
    ax.set_title('Ice')
    plt.colorbar(im,ax=ax,orientation='horizontal')
    
    ax = ax_arr[1]
    im = ax.contour(data_dict['x'][i_time],data_dict['y'][i_time],h2o,cmap='viridis')
    ax.set_title('H2O')
    plt.colorbar(im,ax=ax,orientation='horizontal')

def plot_mass_multi(multidata,var='Ice Mass'):
    f, ax = plt.subplots()
    for sim_name, dataset in multidata.items():
        t_vec = dataset['t']
        ax.plot(t_vec,dataset[var],label=sim_name)
    ax.legend()
    return f, ax

def mass_comparison(multidata,var='Ice Mass',ref='base'):
    n_max=0
    for sim_name, dataset in multidata.items():
        t_vec = dataset['t']
        n_max=max(n_max,len(t_vec))
    n_sims = len(multidata)
    full_data = {}
    for sim_name, dataset in multidata.items():
        full_data[sim_name] = np.zeros(n_max) + np.nan
        n_t = len(dataset['t'])
        full_data[sim_name][:n_t] = dataset[var][:]
    ref_mass = np.nansum(full_data[ref])
    ref_steps = n_max - np.sum(np.isnan(full_data[ref]))
    print(f'REFERENCE INTEGRATED MASS: {ref_mass:0.1f}, lifetime {ref_steps:d}')
    max_err = 0.0
    for sim_name, data_vec in full_data.items():
        if sim_name == ref:
            continue
        n_steps = n_max - np.sum(np.isnan(data_vec))
        mass_err = (np.nansum(data_vec)/ref_mass - 1.0)
        print(f' --> {sim_name:20s}: {mass_err:0.4%} mass difference, lifetime {n_steps:d}')
        max_err=max(max_err,mass_err)
    return max_err

if __name__ == '__main__':
    from sys import argv
    if len(argv) < 2:
        raise ValueError('No pull request ID supplied')
    pr_id = int(argv[1])
    pr_dir=f'test_{pr_id:d}'
    assert os.path.isdir(pr_dir), f'Directory {pr_dir} not found'
    multidata = {}
    for test_name in ['base','updated']:
        dir_path = os.path.join(pr_dir,'Run',test_name)
        out_data = read_apcemm_dir(dir_path,None,var_list=None)
        if out_data is None:
            print('No data for ' + test_name)
            continue
        multidata[test_name] = out_data

    f, ax = plot_mass_multi(multidata)
    max_err = mass_comparison(multidata)
    ax.set_title(f'Max integrated error: {max_err:0.5%}')
    f.savefig(os.path.join(pr_dir,'Mass_Comparison_1D.png'))
    plt.close(f)
    f, ax_arr, im_diff = plot_data_diff('Ice aerosol volume',multidata['base'],multidata['updated'],
                                         n_times=None,clim=None)
    f.savefig(os.path.join(pr_dir,'IceVol_Comparison_2D.png'))
    plt.close(f)
