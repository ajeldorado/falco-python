import falco

def falco_gen_dm_surf(dm,dx,N):
    pass

def falco_gen_EHLC_FPM_surf_from_cube(dm,flagModel):
    pass

def falco_gen_dm_poke_cube(dm, mp, dx_dm, flagGenCube=True, **kwds):
    # SFF NOTE:  This function exists in falco/lib/dm/falco_gen_dm_poke_cube.py but not sure if this is good
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    pass

    return mp.dm1.compact
