"""Control functions for WFSC."""

import numpy as np
# import multiprocessing
# from astropy.io import fits
# import matplotlib.pyplot as plt
import falco


def wrapper(mp, cvar, jacStruct):
    """
    Outermost wrapper function for all the controller functions.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.

    Returns
    -------
    None
        Changes are made by reference to mp.
    """
#    if type(mp) is not falco.config.ModelParameters:
#        raise TypeError('Input "mp" must be of type ModelParameters')
#    pass

    # with falco.util.TicToc('Using the Jacobian to make other matrices'):
    print('Using the Jacobian to make other matrices...', end='')
    
    # Compute matrices for linear control with regular EFC
    cvar.GstarG_wsum = np.zeros((cvar.NeleAll, cvar.NeleAll))
    cvar.RealGstarEab_wsum = np.zeros((cvar.NeleAll, 1))

    for im in range(mp.jac.Nmode):

        Gmode = np.zeros((mp.Fend.corr.Npix, 1), dtype=complex)  # Initialize a row to concatenate onto
        if(any(mp.dm_ind == 1)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G1[:,:,im])))
        if(any(mp.dm_ind == 2)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G2[:,:,im])))
        if(any(mp.dm_ind == 8)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G8[:,:,im])))
        if(any(mp.dm_ind == 9)): Gmode = np.hstack((Gmode,np.squeeze(jacStruct.G9[:,:,im])))
        Gmode = Gmode[:, 1:]  # Remove the zero column used for initialization

        # Square matrix part stays the same if no re-linearization has occurrred.
        cvar.GstarG_wsum += mp.jac.weights[im]*np.real(np.conj(Gmode).T @ Gmode)

        # The G^*E part changes each iteration because the E-field changes.
        # Apply 2-D spatial weighting to E-field in dark hole pixels.
        Eweighted = mp.WspatialVec*cvar.EfieldVec[:, im]
        # Apply the Jacobian weights and add to the total.
        cvar.RealGstarEab_wsum += mp.jac.weights[im]*np.real(
            np.conj(Gmode).T @ Eweighted.reshape(mp.Fend.corr.Npix, 1))
    
    # Make the regularization matrix. (Define only diagonal here to save RAM.)
    cvar.EyeGstarGdiag = np.max(np.diag(cvar.GstarG_wsum))*np.ones(cvar.NeleAll)
    cvar.EyeNorm = np.max(np.diag(cvar.GstarG_wsum))
    print('done.')

    # Call the Controller Function
    print('Control beginning ...')
    # Established, conventional controllers
    if mp.controller.lower() == 'plannedefc':
        dDM = _planned_efc(mp, cvar)
    elif mp.controller.lower() == 'gridsearchefc':
        dDM = _grid_search_efc(mp, cvar)
            
    # Update the DM commands by adding the delta control signal
    if(any(mp.dm_ind == 1)): mp.dm1.V += dDM.dDM1V
    if(any(mp.dm_ind == 2)): mp.dm2.V += dDM.dDM2V
    if(any(mp.dm_ind == 8)): mp.dm8.V += dDM.dDM8V
    if(any(mp.dm_ind == 9)): mp.dm9.V += dDM.dDM9V

    # Save the delta from the previous command
    if(any(mp.dm_ind == 1)): mp.dm1.dV = dDM.dDM1V
    if(any(mp.dm_ind == 2)): mp.dm2.dV = dDM.dDM2V
    if(any(mp.dm_ind == 8)): mp.dm8.dV = dDM.dDM8V
    if(any(mp.dm_ind == 9)): mp.dm9.dV = dDM.dDM9V


def cull_actuators(mp, cvar, jacStruct):
    """
    Remove weak actuators from the controlled set.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    jacStruct : ModelParameters
        Structure containing control Jacobians for each specified DM.

    Returns
    -------
    None
        Changes are made by reference to mp and jacStruct.
    """
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')
    
    # Reduce the number of actuators used based on their relative strength
    # in the Jacobian
    if(cvar.flagCullAct and cvar.flagRelin):
        
        print('Weeding out weak actuators from the control Jacobian...')
        if(any(mp.dm_ind == 1)):
            G1intNorm = np.sum(np.mean(np.abs(jacStruct.G1)**2, axis=2), axis=0)
            G1intNorm = G1intNorm/np.max(G1intNorm)
            mp.dm1.act_ele = np.nonzero(G1intNorm >= 10**(mp.logGmin))[0]
            del G1intNorm
        if(any(mp.dm_ind == 2)):
            G2intNorm = np.sum(np.mean(np.abs(jacStruct.G2)**2, axis=2), axis=0)
            G2intNorm = G2intNorm/np.max(G2intNorm)
            mp.dm2.act_ele = np.nonzero(G2intNorm >= 10**(mp.logGmin))[0]
            del G2intNorm
        if(any(mp.dm_ind == 8)):
            G8intNorm = np.sum(np.mean(np.abs(jacStruct.G8)**2, axis=2), axis=0)
            G8intNorm = G8intNorm/np.max(G8intNorm)
            mp.dm8.act_ele = np.nonzero(G8intNorm >= 10**(mp.logGmin))[0]
            del G8intNorm
        if(any(mp.dm_ind == 9)):
            G9intNorm = np.sum(np.mean(np.abs(jacStruct.G9)**2, axis=2), axis=0)
            G9intNorm = G9intNorm/np.max(G9intNorm)
            mp.dm9.act_ele = np.nonzero(G9intNorm >= 10**(mp.logGmin))[0]
            del G9intNorm

        # Add back in all actuators that are tied (to make the tied actuator
        # logic easier)
        if(any(mp.dm_ind == 1)):
            for ti in range(mp.dm1.tied.shape[0]):
                if not (any(mp.dm1.act_ele == mp.dm1.tied[ti, 0])):
                    mp.dm1.act_ele = np.hstack([mp.dm1.act_ele, mp.dm1.tied[ti, 0]])
                if not (any(mp.dm1.act_ele == mp.dm1.tied[ti, 1])):
                    mp.dm1.act_ele = np.hstack([mp.dm1.act_ele, mp.dm1.tied[ti, 1]])
            # Need to sort for the logic in model_Jacobian.m
            mp.dm1.act_ele = np.sort(mp.dm1.act_ele)

        if(any(mp.dm_ind == 2)):
            for ti in range(mp.dm2.tied.shape[0]):
                if not any(mp.dm2.act_ele == mp.dm2.tied[ti, 0]):
                    mp.dm2.act_ele = np.hstack([mp.dm2.act_ele, mp.dm2.tied[ti, 0]])
                if not any(mp.dm2.act_ele == mp.dm2.tied[ti, 1]):
                    mp.dm2.act_ele = np.hstack([mp.dm2.act_ele, mp.dm2.tied[ti, 1]])
            # Need to sort for the logic in model_Jacobian.m
            mp.dm2.act_ele = np.sort(mp.dm2.act_ele)
#            if(any(mp.dm_ind == 8))
#                for ti=1:size(mp.dm8.tied,1)
#                    if(any(mp.dm8.act_ele==mp.dm8.tied(ti,1))==false);  mp.dm8.act_ele = [mp.dm8.act_ele; mp.dm8.tied(ti,1)];  end
#                    if(any(mp.dm8.act_ele==mp.dm8.tied(ti,2))==false);  mp.dm8.act_ele = [mp.dm8.act_ele; mp.dm8.tied(ti,2)];  end
#                end
#                mp.dm8.act_ele = sort(mp.dm8.act_ele);
#            end
#            if(any(mp.dm_ind == 9))
#                for ti=1:size(mp.dm9.tied,1)
#                    if(any(mp.dm9.act_ele==mp.dm9.tied(ti,1))==false);  mp.dm9.act_ele = [mp.dm9.act_ele; mp.dm9.tied(ti,1)];  end
#                    if(any(mp.dm9.act_ele==mp.dm9.tied(ti,2))==false);  mp.dm9.act_ele = [mp.dm9.act_ele; mp.dm9.tied(ti,2)];  end
#                end
#                mp.dm9.act_ele = sort(mp.dm9.act_ele);
#            end
            
        # Update the number of elements used per DM
        if(any(mp.dm_ind == 1)): mp.dm1.Nele = mp.dm1.act_ele.size
        if(any(mp.dm_ind == 2)): mp.dm2.Nele = mp.dm2.act_ele.size
        if(any(mp.dm_ind == 8)): mp.dm8.Nele = mp.dm8.act_ele.size
        if(any(mp.dm_ind == 9)): mp.dm9.Nele = mp.dm9.act_ele.size

        if(any(mp.dm_ind == 1)): print('  DM1: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm1.Nele, mp.dm1.NactTotal, 100*mp.dm1.Nele/mp.dm1.NactTotal))
        if(any(mp.dm_ind == 2)): print('  DM2: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm2.Nele, mp.dm2.NactTotal, 100*mp.dm2.Nele/mp.dm2.NactTotal))
        if(any(mp.dm_ind == 8)): print('  DM8: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm8.Nele, mp.dm8.NactTotal, 100*mp.dm8.Nele/mp.dm8.NactTotal))
        if(any(mp.dm_ind == 9)): print('  DM9: %d/%d (%.2f%%) actuators kept for Jacobian' % (mp.dm9.Nele, mp.dm9.NactTotal, 100*mp.dm9.Nele/mp.dm9.NactTotal))
        
        # Crop out unused actuators from the control Jacobian
        if(any(mp.dm_ind == 1)): jacStruct.G1 = jacStruct.G1[:, mp.dm1.act_ele, :]
        if(any(mp.dm_ind == 2)): jacStruct.G2 = jacStruct.G2[:, mp.dm2.act_ele, :]
        if(any(mp.dm_ind == 8)): jacStruct.G8 = jacStruct.G8[:, mp.dm8.act_ele, :]
        if(any(mp.dm_ind == 9)): jacStruct.G9 = jacStruct.G9[:, mp.dm9.act_ele, :]

    return None


def _grid_search_efc(mp, cvar):
    """
    Perform a grid search over specified variables for the controller.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """
    # Make all combinations of the values
    vals_list = [(x, y) for y in mp.ctrl.dmfacVec for x in mp.ctrl.log10regVec]
    Nvals = len(mp.ctrl.log10regVec) * len(mp.ctrl.dmfacVec)
    InormVec = np.zeros(Nvals)  # Initialize

    # Temporarily store computed DM commands so that the best one does not have
    # to be re-computed
    if(any(mp.dm_ind == 1)):
        dDM1V_store = np.zeros((mp.dm1.Nact, mp.dm1.Nact, Nvals))
    if(any(mp.dm_ind == 2)):
        dDM2V_store = np.zeros((mp.dm2.Nact, mp.dm2.Nact, Nvals))
    if(any(mp.dm_ind == 8)):
        dDM8V_store = np.zeros((mp.dm8.NactTotal, Nvals))
    if(any(mp.dm_ind == 9)):
        dDM9V_store = np.zeros((mp.dm9.NactTotal, Nvals))

    # Empirically find the regularization value giving the best contrast
#    if(mp.flagMultiproc and mp.ctrl.flagUseModel):
#        # Run the controller in parallel
#        pool = multiprocessing.Pool(processes=mp.Nthreads)
#        results = [pool.apply_async(_efc, args=(ni,vals_list,mp,cvar)) for ni in np.arange(Nvals,dtype=int) ]
#        results_ctrl = [p.get() for p in results] # All the Jacobians in a list
#        pool.close()
#        pool.join()
#
#        # Convert from a list to arrays:
#        for ni in range(Nvals):
#            InormVec[ni] = results_ctrl[ni][0]
#            if(any(mp.dm_ind == 1)): dDM1V_store[:,:,ni] = results_ctrl[ni][1].dDM1V
#            if(any(mp.dm_ind == 2)): dDM2V_store[:,:,ni] = results_ctrl[ni][1].dDM2V
#    else:
    for ni in range(Nvals):
        [InormVec[ni], dDM_temp] = _efc(ni, vals_list, mp, cvar)
        # delta voltage commands
        if(any(mp.dm_ind == 1)): dDM1V_store[:, :, ni] = dDM_temp.dDM1V
        if(any(mp.dm_ind == 2)): dDM2V_store[:, :, ni] = dDM_temp.dDM2V
        if(any(mp.dm_ind == 8)): dDM8V_store[:, ni] = dDM_temp.dDM8V
        if(any(mp.dm_ind == 9)): dDM9V_store[:, ni] = dDM_temp.dDM9V

    # Print out results to the command line
    print('Scaling factor:\t', end='')
    for ni in range(Nvals):
        print('%.2f\t\t' % (vals_list[ni][1]), end='')

    print('\nlog10reg:    \t', end='')
    for ni in range(Nvals):
        print('%.1f\t\t' % (vals_list[ni][0]), end='')
    
    print('\nInorm:       \t', end='')
    for ni in range(Nvals):
        print('%.2e\t' % (InormVec[ni]), end='')
    print('\n', end='')

    # Find the best scaling factor and Lagrange multiplier pair based on the
    # best contrast.
    # [cvar.cMin,indBest] = np.min(InormVec)
    indBest = np.argmin(InormVec)
    cvar.cMin = np.min(InormVec)
    dDM = falco.config.Object()
    # delta voltage commands
    if(any(mp.dm_ind == 1)): dDM.dDM1V = np.squeeze(dDM1V_store[:, :, indBest])
    if(any(mp.dm_ind == 2)): dDM.dDM2V = np.squeeze(dDM2V_store[:, :, indBest])
    if(any(mp.dm_ind == 8)): dDM.dDM8V = np.squeeze(dDM8V_store[:, indBest])
    if(any(mp.dm_ind == 9)): dDM.dDM9V = np.squeeze(dDM9V_store[:, indBest])

    cvar.log10regUsed = vals_list[indBest][0]
    dmfacBest = vals_list[indBest][1]
    if(mp.ctrl.flagUseModel):
        print('Model-based grid search expects log10reg, = %.1f,\t dmfac = %.2f,\t %4.2e normalized intensity.'
              % (cvar.log10regUsed, dmfacBest, cvar.cMin))
    else:
        print('Empirical grid search finds log10reg, = %.1f,\t dmfac = %.2f,\t %4.2e normalized intensity.'
              % (cvar.log10regUsed, dmfacBest, cvar.cMin))
    
    return dDM


def _planned_efc(mp, cvar):
    """
    Perform a scheduled/planned set of EFC iterations.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """
    # Make all combinations of the values
    vals_list = [(x, y) for y in mp.ctrl.dmfacVec for x in mp.ctrl.log10regVec]
    Nvals = len(mp.ctrl.log10regVec) * len(mp.ctrl.dmfacVec)
    InormVec = np.zeros(Nvals)  # Initialize

    # Make more obvious names for conditions:
    relinearizeNow = any(np.array(mp.gridSearchItrVec) == cvar.Itr)
    useBestLog10Reg = np.imag(mp.ctrl.log10regSchedIn[cvar.Itr]) != 0
    realLog10RegIsZero = np.real(mp.ctrl.log10regSchedIn[cvar.Itr]) == 0

    # Step 1: Empirically find the "optimal" regularization value
    # (if told to for this iteration).
    
    if relinearizeNow:
        
        # Temporarily store computed DM commands so that the best one does
        # not have to be re-computed
        if(any(mp.dm_ind == 1)):
            dDM1V_store = np.zeros((mp.dm1.Nact, mp.dm1.Nact, Nvals))
        if(any(mp.dm_ind == 2)):
            dDM2V_store = np.zeros((mp.dm2.Nact, mp.dm2.Nact, Nvals))
        if(any(mp.dm_ind == 8)):
            dDM8V_store = np.zeros((mp.dm8.NactTotal, Nvals))
        if(any(mp.dm_ind == 9)):
            dDM9V_store = np.zeros((mp.dm9.NactTotal, Nvals))
        
        for ni in range(Nvals):
            [InormVec[ni], dDM_temp] = _efc(ni, vals_list, mp, cvar)
            # delta voltage commands
            if(any(mp.dm_ind == 1)): dDM1V_store[:, :, ni] = dDM_temp.dDM1V
            if(any(mp.dm_ind == 2)): dDM2V_store[:, :, ni] = dDM_temp.dDM2V
            if(any(mp.dm_ind == 8)): dDM8V_store[:, ni] = dDM_temp.dDM8V
            if(any(mp.dm_ind == 9)): dDM9V_store[:, ni] = dDM_temp.dDM9V

        # Print out results to the command line
        print('Scaling factor:\t', end='')
        for ni in range(Nvals): print('%.2f\t\t' % (vals_list[ni][1]), end='')
    
        print('\nlog10reg:    \t', end='')
        for ni in range(Nvals): print('%.1f\t\t' % (vals_list[ni][0]), end='')
        
        print('\nInorm:       \t', end='')
        for ni in range(Nvals): print('%.2e\t' % (InormVec[ni]), end='')
        print('\n', end='')
    
        # Find the best scaling factor and Lagrange multiplier pair based on
        # the best contrast.
        # [cvar.cMin,indBest] = np.min(InormVec)
        indBest = np.argmin(InormVec)
        cvar.cMin = np.min(InormVec)
        cvar.latestBestlog10reg = vals_list[indBest][0]
        cvar.latestBestDMfac = vals_list[indBest][1]
        
        if(mp.ctrl.flagUseModel):
            print(('Model-based grid search expects log10reg, = %.1f,\t ' +
                  'dmfac = %.2f,\t %4.2e normalized intensity.') %
                  (cvar.latestBestlog10reg, cvar.latestBestDMfac, cvar.cMin))
        else:
            print(('Empirical grid search finds log10reg, = %.1f,\t dmfac' +
                  ' = %.2f,\t %4.2e normalized intensity.') %
                  (cvar.latestBestlog10reg, cvar.latestBestDMfac, cvar.cMin))
    
    # Skip steps 2 and 3 if the schedule for this iteration is just to use the
    # "optimal" regularization AND if grid search was performed this iteration.
    if relinearizeNow and useBestLog10Reg and realLog10RegIsZero:
        # delta voltage commands
        dDM = falco.config.Object()  # Initialize
        if(any(mp.dm_ind == 1)): dDM.dDM1V = np.squeeze(dDM1V_store[:, :, indBest])
        if(any(mp.dm_ind == 2)): dDM.dDM2V = np.squeeze(dDM2V_store[:, :, indBest])
        if(any(mp.dm_ind == 8)): dDM.dDM8V = np.squeeze(dDM8V_store[:, indBest])
        if(any(mp.dm_ind == 9)): dDM.dDM9V = np.squeeze(dDM9V_store[:, indBest])
        
        log10regSchedOut = cvar.latestBestlog10reg
    else:
        # Step 2: For this iteration in the schedule, replace the imaginary
        # part of the regularization with the latest "optimal" regularization
        if useBestLog10Reg:
            log10regSchedOut = cvar.latestBestlog10reg + \
                np.real(mp.ctrl.log10regSchedIn[cvar.Itr])
        else:
            log10regSchedOut = np.real(mp.ctrl.log10regSchedIn[cvar.Itr])

        # Step 3: Compute the EFC command to use
        ni = 0
        if not hasattr(cvar, 'latestBestDMfac'):
            cvar.latestBestDMfac = 1
        vals_list = [(x, y) for y in np.array([cvar.latestBestDMfac])
                      for x in np.array([log10regSchedOut])]
    
        [cvar.cMin, dDM] = _efc(ni, vals_list, mp, cvar)
        if mp.ctrl.flagUseModel:
            print(('Model expects scheduled log10(reg) = %.1f\t to give ' +
                  '%4.2e normalized intensity.') %
                  (log10regSchedOut, cvar.cMin))
        else:
            print(('Scheduled log10reg = %.1f\t gives %4.2e normalized' +
                  ' intensity.') % (log10regSchedOut, cvar.cMin))
    
    cvar.log10regUsed = log10regSchedOut
    
    return dDM


def efc_schedule_generator(scheduleMatrix):
    """
    Generate the EFC schedule from an input matrix.

    Parameters
    ----------
    scheduleMatrix : array_like
        DESCRIPTION.

    Returns
    -------
    Nitr : int
        Number of WFSC iterations.
    relinItrVec : array_like
        DESCRIPTION.
    gridSearchItrVec : array_like
        DESCRIPTION.
    log10regSched : array_like
        DESCRIPTION.
    dm_ind_sched : list
        DESCRIPTION.

    Notes
    -----
    CONTROL SCHEDULE. Columns of sched_mat are:
    % Column 0: # of iterations,
    % Column 1: log10(regularization),
    % Column 2: which DMs to use (12, 128, 129, or 1289) for control
    % Column 3: flag (0 = false, 1 = true), whether to re-linearize
    %   at that iteration.
    % Column 4: flag (0 = false, 1 = true), whether to perform an
    %   EFC parameter grid search to find the set giving the best
    %   contrast .
    % The imaginary part of the log10(regularization) in column 1 is
    %  replaced for that iteration with the optimal log10(regularization)
    % A row starting with [0, 0, 0, 1...] relinearizes only at that time
    """
    # Number of correction iterations
    Nitr = int(np.real(np.sum(scheduleMatrix[:, 0])))
    
    # Create the vectors of:
    #  1) iteration numbers at which to relinearize the Jacobian
    #  2) log10(regularization) at each correction iteration
    relinItrVec = np.array([])  # Initialize
    gridSearchItrVec = []  # Initialize
    log10regSched = np.zeros((Nitr,), dtype=complex)  # Initialize
    dmIndList = np.zeros((Nitr,), dtype=int)  # Initialize
    iterCount = 0
    for iRow in range(scheduleMatrix.shape[0]):
        
        # When to re-linearize
        if int(np.real(scheduleMatrix[iRow, 3])) == 1:
            relinItrVec = np.append(relinItrVec, iterCount)

        # When to re-do the empirical EFC grid search
        if int(np.real(scheduleMatrix[iRow, 4])) == 1:
            gridSearchItrVec.append(iterCount)
                
        # Make the vector of regularizations at each iteration
        deltaIter = int(np.real(scheduleMatrix[iRow, 0]))
        if not deltaIter == 0:
            log10regSched[iterCount:(iterCount+deltaIter)] = \
                scheduleMatrix[iRow, 1]
            dmIndList[iterCount:(iterCount+deltaIter)] = \
                int(np.real(scheduleMatrix[iRow, 2]))

        iterCount += int(np.real(scheduleMatrix[iRow, 0]))

    gridSearchItrVec = np.asarray(gridSearchItrVec)

    # Store DM number index vectors as cells since they can vary in length
    dm_ind_sched = []
    for Itr in range(Nitr):
        dm_ind = []
        numAsStr = str(dmIndList[Itr])
        for ii in range(len(numAsStr)):
            dm_ind.append(int(numAsStr[ii]))
        dm_ind_sched.append(np.array(dm_ind))

    return Nitr, relinItrVec, gridSearchItrVec, log10regSched, dm_ind_sched


def _efc(ni, vals_list, mp, cvar):
    """
    Compute the main EFC equation. Called by a wrapper controller function.

    Parameters
    ----------
    ni : int
        index for the set of possible combinations of variables to do a grid
        search over
    vals_list : list
        the set of possible combinations of values to do a grid search over
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    InormAvg : float
        Normalized intensity averaged spectrally and spatially.
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """
    #  function [InormAvg,dDM] = _efc(ni,vals_list,mp,cvar)

    if(any(mp.dm_ind == 1)):
        DM1V0 = mp.dm1.V.copy()
    if(any(mp.dm_ind == 2)):
        DM2V0 = mp.dm2.V.copy()
    if(any(mp.dm_ind == 8)):
        DM8V0 = mp.dm8.V.copy()
    if(any(mp.dm_ind == 9)):
        DM9V0 = mp.dm9.V.copy()
        
    # Initializations
    log10reg = vals_list[ni][0]  # log 10 of regularization value
    dmfac = vals_list[ni][1]  # Scaling factor for entire DM command
    
    # Save starting point for each delta command to be added to.
    # Get the indices of each DM's command vector within the single
    # concatenated command vector
    init(mp, cvar)  # Modifies cvar
    
    # Least-squares solution with regularization:
    duVecNby1 = \
        -dmfac*np.linalg.solve((10**log10reg*np.diag(cvar.EyeGstarGdiag) +
                                cvar.GstarG_wsum), cvar.RealGstarEab_wsum)
    # Convert to true 1-D array from a 2-D, Nx1 array
    duVec = duVecNby1.reshape((-1,))
     
    # Parse the command vector by DM and assign the output commands
    mp, dDM = wrapup(mp, cvar, duVec)
    
    # Take images and compute average intensity in dark hole
    if(mp.ctrl.flagUseModel):  # Get simulated image from compact model
        Itotal = falco.imaging.get_expected_summed_image(mp, cvar, dDM)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
    else:  # Get actual image from full model or testbed
        Itotal = falco.imaging.get_summed_image(mp)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
        
    # Reset voltage commands in mp
    if(any(mp.dm_ind == 1)):
        mp.dm1.V = DM1V0
    if(any(mp.dm_ind == 2)):
        mp.dm2.V = DM2V0
    if(any(mp.dm_ind == 8)):
        mp.dm8.V = DM8V0
    if(any(mp.dm_ind == 9)):
        mp.dm9.V = DM9V0
        
    return InormAvg, dDM


def init(mp, cvar):
    """
    Vectorize DM commands and otherwise prepare variables for the controller.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables

    Returns
    -------
    None
        Changes are made by reference to structures mp and cvar.
    """
    # Save starting point for each delta command to be added to.
    if(any(mp.dm_ind == 1)): cvar.DM1Vnom = mp.dm1.V
    if(any(mp.dm_ind == 2)): cvar.DM2Vnom = mp.dm2.V
    if(any(mp.dm_ind == 8)): cvar.DM8Vnom = mp.dm8.V
    if(any(mp.dm_ind == 9)): cvar.DM9Vnom = mp.dm9.V
    
    # Make the vector of total DM commands from before
    u1 = mp.dm1.V.reshape(mp.dm1.NactTotal)[mp.dm1.act_ele] if(any(mp.dm_ind == 1)) else np.array([])
    u2 = mp.dm2.V.reshape(mp.dm2.NactTotal)[mp.dm2.act_ele] if(any(mp.dm_ind == 2)) else np.array([])
    u8 = mp.dm9.V.reshape(mp.dm8.NactTotal)[mp.dm8.act_ele] if(any(mp.dm_ind == 8)) else np.array([])
    u9 = mp.dm9.V.reshape(mp.dm9.NactTotal)[mp.dm9.act_ele] if(any(mp.dm_ind == 9)) else np.array([])
    cvar.uVec = np.concatenate((u1, u2, u8, u9))
    cvar.NeleAll = cvar.uVec.size
    
    # Get the indices of each DM's command within the full command
    u1dummy = 1*np.ones(mp.dm1.Nele, dtype=int) if(any(mp.dm_ind == 1)) else np.array([])
    u2dummy = 2*np.ones(mp.dm2.Nele, dtype=int) if(any(mp.dm_ind == 2)) else np.array([])
    u8dummy = 8*np.ones(mp.dm8.Nele, dtype=int) if(any(mp.dm_ind == 8)) else np.array([])
    u9dummy = 9*np.ones(mp.dm9.Nele, dtype=int) if(any(mp.dm_ind == 9)) else np.array([])
    cvar.uLegend = np.concatenate((u1dummy, u2dummy, u8dummy, u9dummy))
    

def wrapup(mp, cvar, duVec):
    """
    Apply controller commands to the DM command variables.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    cvar : ModelParameters
        Structure containing controller variables
    duVec : numpy ndarray
        Vector of delta control commands computed by the controller.

    Returns
    -------
    mp : ModelParameters
        Structure containing optical model parameters
    dDM : ModelParameters
        Structure containing the delta DM commands for each DM
    """
    dDM = falco.config.Object()
    
    # Parse the command vector by DM and apply weighting
    if(any(mp.dm_ind == 1)):
        dDM1Vvec = np.zeros(mp.dm1.NactTotal)
        dDM1Vvec[mp.dm1.act_ele] = mp.dm1.weight*duVec[cvar.uLegend == 1]
        dDM.dDM1V = dDM1Vvec.reshape((mp.dm1.Nact, mp.dm1.Nact))
    if(any(mp.dm_ind == 2)):
        dDM2Vvec = np.zeros(mp.dm2.NactTotal)
        dDM2Vvec[mp.dm2.act_ele] = mp.dm2.weight*duVec[cvar.uLegend == 2]
        dDM.dDM2V = dDM2Vvec.reshape((mp.dm2.Nact, mp.dm2.Nact))
    if(any(mp.dm_ind == 8)):
        dDM.dDM8V = np.zeros(mp.dm8.NactTotal)
        dDM.dDM8V[mp.dm8.act_ele] = mp.dm8.weight*duVec[cvar.uLegend == 8]
    if(any(mp.dm_ind == 9)):
        dDM.dDM9V = np.zeros(mp.dm9.NactTotal)
        dDM.dDM9V[mp.dm9.act_ele] = mp.dm9.weight*duVec[cvar.uLegend == 9]
    
    # Enforce tied actuator pair commands.
    #   Assign command of first actuator to the second as well.
#    if(any(mp.dm_ind == 8)):
#        for ti=1:size(mp.dm8.tied,1)
#            dDM.dDM8V(mp.dm8.tied(ti,2)) = dDM.dDM8V(mp.dm8.tied(ti,1));
#
#    if(any(mp.dm_ind == 9)):
#        for ti=1:size(mp.dm9.tied,1)
#            dDM.dDM9V(mp.dm9.tied(ti,2)) = dDM.dDM9V(mp.dm9.tied(ti,1));

    # Combine the delta command with the previous command
    if(any(mp.dm_ind == 1)): mp.dm1.V = cvar.DM1Vnom + dDM.dDM1V
    if(any(mp.dm_ind == 2)): mp.dm2.V = cvar.DM2Vnom + dDM.dDM2V
    if(any(mp.dm_ind == 8)): mp.dm8.V = cvar.DM8Vnom + dDM.dDM8V
    if(any(mp.dm_ind == 9)): mp.dm9.V = cvar.DM9Vnom + dDM.dDM9V

    return mp, dDM
