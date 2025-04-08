"""Control functions for WFSC."""
import copy
import time

from concurrent.futures import ThreadPoolExecutor as PoolExecutor
# from concurrent.futures import ProcessPoolExecutor as PoolExecutor
import multiprocessing
import numpy as np
import scipy.optimize

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
    if type(mp) is not falco.config.ModelParameters:
        raise TypeError('Input "mp" must be of type ModelParameters')

    if mp.controller.lower() == 'ad-efc':

        # Call the Controller Function
        print('Control beginning ...')
        dDM = _planned_ad_efc(mp, cvar)

    else:

        apply_spatial_weighting_to_Jacobian(mp, jacStruct)
    
        # with falco.util.TicToc('Using the Jacobian to make other matrices'):
        print('Using the Jacobian to make other matrices...', end='')
    
        # Compute matrices for linear control with regular EFC
        cvar.GstarG_wsum = np.zeros((cvar.NeleAll, cvar.NeleAll))
        cvar.RealGstarEab_wsum = np.zeros((cvar.NeleAll, 1))
        Eest = cvar.Eest.copy()
    
        for iMode in range(mp.jac.Nmode):
    
            Gstack = np.zeros((mp.Fend.corr.Npix, 1), dtype=complex)  # Initialize a row to concatenate onto
            if any(mp.dm_ind == 1):
                Gstack = np.hstack((Gstack, np.squeeze(jacStruct.G1[:, :, iMode])))
            if any(mp.dm_ind == 2):
                Gstack = np.hstack((Gstack, np.squeeze(jacStruct.G2[:, :, iMode])))
            if any(mp.dm_ind == 8):
                Gstack = np.hstack((Gstack, np.squeeze(jacStruct.G8[:, :, iMode])))
            if any(mp.dm_ind == 9):
                Gstack = np.hstack((Gstack, np.squeeze(jacStruct.G9[:, :, iMode])))
            Gstack = Gstack[:, 1:]  # Remove the column used for initialization
    
            # Square matrix part stays the same if no re-linearization has occurrred.
            cvar.GstarG_wsum += mp.jac.weights[iMode]*np.real(np.conj(Gstack).T @ Gstack)
    
            if mp.jac.minimizeNI:
                modvar = falco.config.ModelVariables()
                modvar.whichSource = 'star'
                modvar.sbpIndex = mp.jac.sbp_inds[iMode]
                modvar.zernIndex = mp.jac.zern_inds[iMode]
                modvar.starIndex = mp.jac.star_inds[iMode]
                Eunocculted = falco.model.compact(mp, modvar, useFPM=False)
                indPeak = np.unravel_index(
                    np.argmax(np.abs(Eunocculted), axis=None), Eunocculted.shape)
                Epeak = Eunocculted[indPeak]
                Eest[:, iMode] = cvar.Eest[:, iMode] / Epeak
    
            # The G^*E part changes each iteration because the E-field changes.
            # Apply 2-D spatial weighting to E-field in dark hole pixels.
            iStar = mp.jac.star_inds[iMode]
            Eweighted = mp.WspatialVec[:, iStar] * Eest[:, iMode]
            # Apply the Jacobian weights and add to the total.
            cvar.RealGstarEab_wsum += mp.jac.weights[iMode]*np.real(
                np.conj(Gstack).T @ Eweighted.reshape(mp.Fend.corr.Npix, 1))
    
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
    if any(mp.dm_ind == 1):
        mp.dm1.V += dDM.dDM1V
    if any(mp.dm_ind == 2):
        mp.dm2.V += dDM.dDM2V
    if any(mp.dm_ind == 8):
        mp.dm8.V += dDM.dDM8V
    if any(mp.dm_ind == 9):
        mp.dm9.V += dDM.dDM9V

    # Save the delta from the previous command
    if any(mp.dm_ind == 1):
        mp.dm1.dV = dDM.dDM1V
    if any(mp.dm_ind == 2):
        mp.dm2.dV = dDM.dDM2V
    if any(mp.dm_ind == 8):
        mp.dm8.dV = dDM.dDM8V
    if any(mp.dm_ind == 9):
        mp.dm9.dV = dDM.dDM9V

    # falco_ctrl_update_dm_commands(mp, dDM);

    return None


def cull_weak_actuators(mp, cvar, jacStruct):
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

    if cvar.Itr == 0:  # Cull in first iteration
        cvar.flagCullAct = True
    else:  # Cull when actuators used change
        if hasattr(mp, 'dm_ind_sched'):
            schedPre = np.sort(mp.dm_ind_sched[cvar.Itr-1])
            schedNow = np.sort(mp.dm_ind_sched[cvar.Itr])
            cvar.flagCullAct = not np.array_equal(schedPre, schedNow)
        else:
            cvar.flagCullAct = False

    if not hasattr(mp, 'flagCullActHist'):
        mp.flagCullActHist = np.zeros((mp.Nitr, ))
    mp.flagCullActHist[cvar.Itr] = cvar.flagCullAct

    # Before performing new cull, include all actuators again
    if cvar.flagCullAct:
        # Re-include all actuators in the basis set.
        if np.any(mp.dm_ind == 1):
            mp.dm1.act_ele = list(range(mp.dm1.NactTotal))
        if np.any(mp.dm_ind == 2):
            mp.dm2.act_ele = list(range(mp.dm2.NactTotal))
        if np.any(mp.dm_ind == 8):
            mp.dm8.act_ele = list(range(mp.dm8.NactTotal))
        if np.any(mp.dm_ind == 9):
            mp.dm9.act_ele = list(range(mp.dm9.NactTotal))

        # Update the number of elements used per DM
        if np.any(mp.dm_ind == 1):
            mp.dm1.Nele = len(mp.dm1.act_ele)
        else:
            mp.dm1.Nele = 0
        if np.any(mp.dm_ind == 2):
            mp.dm2.Nele = len(mp.dm2.act_ele)
        else:
            mp.dm2.Nele = 0
        if np.any(mp.dm_ind == 8):
            mp.dm8.Nele = len(mp.dm8.act_ele)
        else:
            mp.dm8.Nele = 0
        if np.any(mp.dm_ind == 9):
            mp.dm9.Nele = len(mp.dm9.act_ele)
        else:
            mp.dm9.Nele = 0

    # Reduce the number of actuators used based on their relative strength
    # in the Jacobian
    if cvar.flagCullAct and cvar.flagRelin:

        print('Weeding out weak actuators from the control Jacobian...')
        if any(mp.dm_ind == 1):
            G1intNorm = np.sum(np.mean(np.abs(jacStruct.G1)**2, axis=2), axis=0)
            G1intNorm = G1intNorm/np.max(G1intNorm)
            mp.dm1.act_ele = np.nonzero(G1intNorm >= 10**(mp.logGmin))[0]
            del G1intNorm
        if any(mp.dm_ind == 2):
            G2intNorm = np.sum(np.mean(np.abs(jacStruct.G2)**2, axis=2), axis=0)
            G2intNorm = G2intNorm/np.max(G2intNorm)
            mp.dm2.act_ele = np.nonzero(G2intNorm >= 10**(mp.logGmin))[0]
            del G2intNorm
        if any(mp.dm_ind == 8):
            G8intNorm = np.sum(np.mean(np.abs(jacStruct.G8)**2, axis=2), axis=0)
            G8intNorm = G8intNorm/np.max(G8intNorm)
            mp.dm8.act_ele = np.nonzero(G8intNorm >= 10**(mp.logGmin))[0]
            del G8intNorm
        if any(mp.dm_ind == 9):
            G9intNorm = np.sum(np.mean(np.abs(jacStruct.G9)**2, axis=2), axis=0)
            G9intNorm = G9intNorm/np.max(G9intNorm)
            mp.dm9.act_ele = np.nonzero(G9intNorm >= 10**(mp.logGmin))[0]
            del G9intNorm

        # Add back in all actuators that are tied (to make the tied actuator
        # logic easier)
        if any(mp.dm_ind == 1):
            for ti in range(mp.dm1.tied.shape[0]):
                if not (any(mp.dm1.act_ele == mp.dm1.tied[ti, 0])):
                    mp.dm1.act_ele = np.hstack([mp.dm1.act_ele, mp.dm1.tied[ti, 0]])
                if not (any(mp.dm1.act_ele == mp.dm1.tied[ti, 1])):
                    mp.dm1.act_ele = np.hstack([mp.dm1.act_ele, mp.dm1.tied[ti, 1]])
            # Need to sort for the logic in model_Jacobian.m
            mp.dm1.act_ele = np.sort(mp.dm1.act_ele)

        if any(mp.dm_ind == 2):
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
        if any(mp.dm_ind == 1):
            mp.dm1.Nele = mp.dm1.act_ele.size
        if any(mp.dm_ind == 2):
            mp.dm2.Nele = mp.dm2.act_ele.size
        if any(mp.dm_ind == 8):
            mp.dm8.Nele = mp.dm8.act_ele.size
        if any(mp.dm_ind == 9):
            mp.dm9.Nele = mp.dm9.act_ele.size

        if any(mp.dm_ind == 1):
            print('  DM1: %d/%d (%.2f%%) actuators kept for Jacobian' %
                  (mp.dm1.Nele, mp.dm1.NactTotal, 100*mp.dm1.Nele/mp.dm1.NactTotal))
        if any(mp.dm_ind == 2):
            print('  DM2: %d/%d (%.2f%%) actuators kept for Jacobian' %
                  (mp.dm2.Nele, mp.dm2.NactTotal, 100*mp.dm2.Nele/mp.dm2.NactTotal))
        if any(mp.dm_ind == 8):
            print('  DM8: %d/%d (%.2f%%) actuators kept for Jacobian' %
                  (mp.dm8.Nele, mp.dm8.NactTotal, 100*mp.dm8.Nele/mp.dm8.NactTotal))
        if any(mp.dm_ind == 9):
            print('  DM9: %d/%d (%.2f%%) actuators kept for Jacobian' %
                  (mp.dm9.Nele, mp.dm9.NactTotal, 100*mp.dm9.Nele/mp.dm9.NactTotal))

        # Crop out unused actuators from the control Jacobian
        if any(mp.dm_ind == 1):
            jacStruct.G1 = jacStruct.G1[:, mp.dm1.act_ele, :]
        if any(mp.dm_ind == 2):
            jacStruct.G2 = jacStruct.G2[:, mp.dm2.act_ele, :]
        if any(mp.dm_ind == 8):
            jacStruct.G8 = jacStruct.G8[:, mp.dm8.act_ele, :]
        if any(mp.dm_ind == 9):
            jacStruct.G9 = jacStruct.G9[:, mp.dm9.act_ele, :]

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
    InormVec = np.zeros(Nvals)
    ImCube = np.zeros((Nvals, mp.Fend.Neta, mp.Fend.Nxi))

    # Temporarily store computed DM commands so that the best one does not have
    # to be re-computed
    if any(mp.dm_ind == 1):
        dDM1V_store = np.zeros((mp.dm1.Nact, mp.dm1.Nact, Nvals))
    if any(mp.dm_ind == 2):
        dDM2V_store = np.zeros((mp.dm2.Nact, mp.dm2.Nact, Nvals))
    if any(mp.dm_ind == 8):
        dDM8V_store = np.zeros((mp.dm8.NactTotal, Nvals))
    if any(mp.dm_ind == 9):
        dDM9V_store = np.zeros((mp.dm9.NactTotal, Nvals))

    # Empirically find the regularization value giving the best contrast

    # Run the controller in parallel only when mp.ctrl.flagUseModel is True because that makes
    # single calls to the compact model. When it is False and in simulation, it calls
    # falco.imaging.get_summed_image(), which has its own internal parallelization.
    if mp.flagParallel and mp.ctrl.flagUseModel:
        # pool = multiprocessing.Pool(processes=mp.Nthreads)
        # results = [pool.apply_async(_efc, args=(ni,vals_list,mp,cvar)) for ni in np.arange(Nvals,dtype=int) ]
        # results_ctrl = [p.get() for p in results] # All the Jacobians in a list
        # pool.close()
        # pool.join()

        pool = multiprocessing.Pool(processes=mp.Nthreads)
        results = pool.starmap(
            _efc, [(ni, vals_list, mp, cvar) for ni in range(Nvals)]
        )
        results_ctrl = results
        pool.close()
        pool.join()

        # with PoolExecutor(max_workers=mp.Nthreads) as executor:
        #     resultsRaw = executor.map(
        #         lambda p: _efc(*p),
        #         [(ni, vals_list, mp, cvar) for ni in range(Nvals)],
        #     )
        # results_ctrl = tuple(resultsRaw)

        # Convert from a list to arrays:
        for ni in range(Nvals):
            InormVec[ni] = results_ctrl[ni][0]
            if any(mp.dm_ind == 1):
                dDM1V_store[:, :, ni] = results_ctrl[ni][1].dDM1V
            if any(mp.dm_ind == 2):
                dDM2V_store[:, :, ni] = results_ctrl[ni][1].dDM2V
            if any(mp.dm_ind == 8):
                dDM8V_store[:, ni] = results_ctrl[ni][1].dDM8V
            if any(mp.dm_ind == 9):
                dDM9V_store[:, ni] = results_ctrl[ni][1].dDM9V
    else:
        for ni in range(Nvals):
            [InormVec[ni], dDM_temp] = _efc(ni, vals_list, mp, cvar)
            ImCube[ni, :, :] = dDM_temp.Itotal
            # delta voltage commands
            if any(mp.dm_ind == 1):
                dDM1V_store[:, :, ni] = dDM_temp.dDM1V
            if any(mp.dm_ind == 2):
                dDM2V_store[:, :, ni] = dDM_temp.dDM2V
            if any(mp.dm_ind == 8):
                dDM8V_store[:, ni] = dDM_temp.dDM8V
            if any(mp.dm_ind == 9):
                dDM9V_store[:, ni] = dDM_temp.dDM9V

    # Print out results to the command line
    print('Scaling factor:\t\t', end='')
    for ni in range(Nvals):
        print('%.2f\t\t' % (vals_list[ni][1]), end='')

    print('\nlog10reg:    \t\t', end='')
    for ni in range(Nvals):
        print('%.1f\t\t' % (vals_list[ni][0]), end='')

    print('\nInorm:       \t\t', end='')
    for ni in range(Nvals):
        print('%.2e\t' % (InormVec[ni]), end='')
    print('\n', end='')

    # Find the best scaling factor and regularization pair based on the
    # best contrast.
    cvar.InormVec = InormVec
    indBest = np.argmin(InormVec)
    cvar.cMin = np.min(InormVec)
    cvar.Im = np.squeeze(ImCube[indBest, :, :])

    # delta voltage commands
    dDM = falco.config.Object()
    if any(mp.dm_ind == 1):
        dDM.dDM1V = np.squeeze(dDM1V_store[:, :, indBest])
    if any(mp.dm_ind == 2):
        dDM.dDM2V = np.squeeze(dDM2V_store[:, :, indBest])
    if any(mp.dm_ind == 8):
        dDM.dDM8V = np.squeeze(dDM8V_store[:, indBest])
    if any(mp.dm_ind == 9):
        dDM.dDM9V = np.squeeze(dDM9V_store[:, indBest])

    cvar.log10regUsed = vals_list[indBest][0]
    dmfacBest = vals_list[indBest][1]
    if mp.ctrl.flagUseModel:
        print('Model-based grid search expects log10reg, = %.1f,\t '
              'dmfac = %.2f,\t %4.2e normalized intensity.'
              % (cvar.log10regUsed, dmfacBest, cvar.cMin))
    else:
        print('Empirical grid search finds log10reg, = %.1f,\t '
              'dmfac = %.2f,\t %4.2e normalized intensity.'
              % (cvar.log10regUsed, dmfacBest, cvar.cMin))

    return dDM


def _planned_ad_efc(mp, cvar):
    """
    Perform a scheduled/planned set of AD EFC iterations.

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
    InormVec = np.zeros(Nvals)

    # # Make more obvious names for conditions:
    # runNewGridSearch = any(np.array(mp.gridSearchItrVec) == cvar.Itr)
    # useBestLog10Reg = np.imag(mp.ctrl.log10regSchedIn[cvar.Itr]) != 0
    # realLog10RegIsZero = np.real(mp.ctrl.log10regSchedIn[cvar.Itr]) == 0

    # Step 1: Empirically find the "optimal" regularization value

    # Temporarily store computed DM commands so that the best one does
    # not have to be re-computed
    if any(mp.dm_ind == 1):
        dDM1V_store = np.zeros((mp.dm1.Nact, mp.dm1.Nact, Nvals))
    if any(mp.dm_ind == 2):
        dDM2V_store = np.zeros((mp.dm2.Nact, mp.dm2.Nact, Nvals))
    if any(mp.dm_ind == 8):
        dDM8V_store = np.zeros((mp.dm8.NactTotal, Nvals))
    if any(mp.dm_ind == 9):
        dDM9V_store = np.zeros((mp.dm9.NactTotal, Nvals))

    ImCube = np.zeros((Nvals, mp.Fend.Neta, mp.Fend.Nxi))

    for ni in range(Nvals):

        [InormVec[ni], dDM_temp] = _ad_efc(ni, vals_list, mp, cvar)
        ImCube[ni, :, :] = dDM_temp.Itotal

        # delta voltage commands
        if any(mp.dm_ind == 1):
            dDM1V_store[:, :, ni] = dDM_temp.dDM1V
        if any(mp.dm_ind == 2):
            dDM2V_store[:, :, ni] = dDM_temp.dDM2V
        if any(mp.dm_ind == 8):
            dDM8V_store[:, ni] = dDM_temp.dDM8V
        if any(mp.dm_ind == 9):
            dDM9V_store[:, ni] = dDM_temp.dDM9V

    # Print out results to the command line
    print('Scaling factor:\t\t', end='')
    for ni in range(Nvals):
        print('%.2f\t\t' % (vals_list[ni][1]), end='')

    print('\nlog10reg:    \t\t', end='')
    for ni in range(Nvals):
        print('%.1f\t\t' % (vals_list[ni][0]), end='')

    print('\nInorm:       \t\t', end='')
    for ni in range(Nvals):
        print('%.2e\t' % (InormVec[ni]), end='')
    print('\n', end='')

    # Find the best scaling factor and Lagrange multiplier pair based on
    # the best contrast.
    # [cvar.cMin,indBest] = np.min(InormVec)
    indBest = np.argmin(InormVec)
    cvar.cMin = np.min(InormVec)
    cvar.Im = np.squeeze(ImCube[indBest, :, :])
    cvar.latestBestlog10reg = vals_list[indBest][0]
    cvar.latestBestDMfac = vals_list[indBest][1]

    if mp.ctrl.flagUseModel:
        print(('Model-based grid search expects log10reg, = %.1f,\t ' +
              'dmfac = %.2f,\t %4.2e normalized intensity.') %
              (cvar.latestBestlog10reg, cvar.latestBestDMfac, cvar.cMin))
    else:
        print(('Empirical grid search finds log10reg, = %.1f,\t dmfac' +
              ' = %.2f,\t %4.2e normalized intensity.') %
              (cvar.latestBestlog10reg, cvar.latestBestDMfac, cvar.cMin))

    # delta voltage commands
    dDM = falco.config.Object()  # Initialize
    if any(mp.dm_ind == 1):
        dDM.dDM1V = np.squeeze(dDM1V_store[:, :, indBest])
    if any(mp.dm_ind == 2):
        dDM.dDM2V = np.squeeze(dDM2V_store[:, :, indBest])
    if any(mp.dm_ind == 8):
        dDM.dDM8V = np.squeeze(dDM8V_store[:, indBest])
    if any(mp.dm_ind == 9):
        dDM.dDM9V = np.squeeze(dDM9V_store[:, indBest])

    log10regSchedOut = cvar.latestBestlog10reg

    cvar.log10regUsed = log10regSchedOut

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
    InormVec = np.zeros(Nvals)

    # Make more obvious names for conditions:
    runNewGridSearch = any(np.array(mp.gridSearchItrVec) == cvar.Itr)
    useBestLog10Reg = np.imag(mp.ctrl.log10regSchedIn[cvar.Itr]) != 0
    realLog10RegIsZero = np.real(mp.ctrl.log10regSchedIn[cvar.Itr]) == 0

    # Step 1: Empirically find the "optimal" regularization value
    # (if told to for this iteration).

    if runNewGridSearch:

        # Temporarily store computed DM commands so that the best one does
        # not have to be re-computed
        if any(mp.dm_ind == 1):
            dDM1V_store = np.zeros((mp.dm1.Nact, mp.dm1.Nact, Nvals))
        if any(mp.dm_ind == 2):
            dDM2V_store = np.zeros((mp.dm2.Nact, mp.dm2.Nact, Nvals))
        if any(mp.dm_ind == 8):
            dDM8V_store = np.zeros((mp.dm8.NactTotal, Nvals))
        if any(mp.dm_ind == 9):
            dDM9V_store = np.zeros((mp.dm9.NactTotal, Nvals))

        ImCube = np.zeros((Nvals, mp.Fend.Neta, mp.Fend.Nxi))

        # # Don't use because _efc() also calls multiprocessing, and there
        # # can't be nested parallelization calls.
        # if mp.flagParallel and (mp.flagSim or mp.ctrl.flagUseModel):

        #     pool = multiprocessing.Pool(processes=mp.Nthreads)
        #     results_ctrl = pool.starmap(
        #         _efc, [(ni, vals_list, mp, cvar) for ni in range(Nvals)]
        #     )
        #     pool.close()
        #     pool.join()

        #     # Convert from a list to arrays:
        #     for ni in range(Nvals):
        #         InormVec[ni] = results_ctrl[ni][0]
        #         if any(mp.dm_ind == 1):
        #             dDM1V_store[:, :, ni] = results_ctrl[ni][1].dDM1V
        #         if any(mp.dm_ind == 2):
        #             dDM2V_store[:, :, ni] = results_ctrl[ni][1].dDM2V
        #         if any(mp.dm_ind == 8):
        #             dDM8V_store[:, ni] = results_ctrl[ni][1].dDM8V
        #         if any(mp.dm_ind == 9):
        #             dDM9V_store[:, ni] = results_ctrl[ni][1].dDM9V

        # else:

        for ni in range(Nvals):

            [InormVec[ni], dDM_temp] = _efc(ni, vals_list, mp, cvar)
            ImCube[ni, :, :] = dDM_temp.Itotal

            # delta voltage commands
            if any(mp.dm_ind == 1):
                dDM1V_store[:, :, ni] = dDM_temp.dDM1V
            if any(mp.dm_ind == 2):
                dDM2V_store[:, :, ni] = dDM_temp.dDM2V
            if any(mp.dm_ind == 8):
                dDM8V_store[:, ni] = dDM_temp.dDM8V
            if any(mp.dm_ind == 9):
                dDM9V_store[:, ni] = dDM_temp.dDM9V

        # Print out results to the command line
        print('Scaling factor:\t\t', end='')
        for ni in range(Nvals):
            print('%.2f\t\t' % (vals_list[ni][1]), end='')

        print('\nlog10reg:    \t\t', end='')
        for ni in range(Nvals):
            print('%.1f\t\t' % (vals_list[ni][0]), end='')

        print('\nInorm:       \t\t', end='')
        for ni in range(Nvals):
            print('%.2e\t' % (InormVec[ni]), end='')
        print('\n', end='')

        # Find the best scaling factor and Lagrange multiplier pair based on
        # the best contrast.
        # [cvar.cMin,indBest] = np.min(InormVec)
        indBest = np.argmin(InormVec)
        cvar.cMin = np.min(InormVec)
        cvar.Im = np.squeeze(ImCube[indBest, :, :])
        cvar.latestBestlog10reg = vals_list[indBest][0]
        cvar.latestBestDMfac = vals_list[indBest][1]

        if mp.ctrl.flagUseModel:
            print(('Model-based grid search expects log10reg, = %.1f,\t ' +
                  'dmfac = %.2f,\t %4.2e normalized intensity.') %
                  (cvar.latestBestlog10reg, cvar.latestBestDMfac, cvar.cMin))
        else:
            print(('Empirical grid search finds log10reg, = %.1f,\t dmfac' +
                  ' = %.2f,\t %4.2e normalized intensity.') %
                  (cvar.latestBestlog10reg, cvar.latestBestDMfac, cvar.cMin))

    # Skip steps 2 and 3 if the schedule for this iteration is just to use the
    # "optimal" regularization AND if grid search was performed this iteration.
    if runNewGridSearch and useBestLog10Reg and realLog10RegIsZero:
        # delta voltage commands
        dDM = falco.config.Object()  # Initialize
        if any(mp.dm_ind == 1):
            dDM.dDM1V = np.squeeze(dDM1V_store[:, :, indBest])
        if any(mp.dm_ind == 2):
            dDM.dDM2V = np.squeeze(dDM2V_store[:, :, indBest])
        if any(mp.dm_ind == 8):
            dDM.dDM8V = np.squeeze(dDM8V_store[:, indBest])
        if any(mp.dm_ind == 9):
            dDM.dDM9V = np.squeeze(dDM9V_store[:, indBest])

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
        cvar.Im = np.squeeze(dDM.Itotal)
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
    if any(mp.dm_ind == 1):
        DM1V0 = mp.dm1.V.copy()
    if any(mp.dm_ind == 2):
        DM2V0 = mp.dm2.V.copy()
    if any(mp.dm_ind == 8):
        DM8V0 = mp.dm8.V.copy()
    if any(mp.dm_ind == 9):
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
        -dmfac*np.linalg.solve((10.0**log10reg*np.diag(cvar.EyeGstarGdiag) +
                                cvar.GstarG_wsum), cvar.RealGstarEab_wsum)
    # Convert to true 1-D array from a 2-D, Nx1 array
    duVec = duVecNby1.reshape((-1,))

    # Parse the command vector by DM and assign the output commands
    mp, dDM = wrapup(mp, cvar, duVec)

    # Take images and compute average intensity in dark hole
    if mp.ctrl.flagUseModel:  # Get simulated image from compact model
        Itotal = falco.imaging.get_expected_summed_image(mp, cvar, dDM)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
    else:  # Get actual image from full model or testbed
        Itotal = falco.imaging.get_summed_image(mp)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
    dDM.Itotal = Itotal

    # Reset voltage commands in mp
    if any(mp.dm_ind == 1):
        mp.dm1.V = DM1V0
    if any(mp.dm_ind == 2):
        mp.dm2.V = DM2V0
    if any(mp.dm_ind == 8):
        mp.dm8.V = DM8V0
    if any(mp.dm_ind == 9):
        mp.dm9.V = DM9V0

    return InormAvg, dDM


def _ad_efc(ni, vals_list, mp, cvar):
    """
    Use algorithmic differentiation to suppress the E-field. 
    
    Called by a wrapper controller function.

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
    if any(mp.dm_ind == 1):
        DM1V0 = mp.dm1.V.copy()
    if any(mp.dm_ind == 2):
        DM2V0 = mp.dm2.V.copy()
    if any(mp.dm_ind == 8):
        DM8V0 = mp.dm8.V.copy()
    if any(mp.dm_ind == 9):
        DM9V0 = mp.dm9.V.copy()

    # Initializations
    log10reg = vals_list[ni][0]  # log 10 of regularization value
    dmfac = vals_list[ni][1]  # Scaling factor for entire DM command

    # Save starting point for each delta command to be added to.
    # Get the indices of each DM's command vector within the single
    # concatenated command vector
    init(mp, cvar)  # Modifies cvar

    # # Least-squares solution with regularization:
    # duVecNby1 = \
    #     -dmfac*np.linalg.solve((10.0**log10reg*np.diag(cvar.EyeGstarGdiag) +
    #                             cvar.GstarG_wsum), cvar.RealGstarEab_wsum)
    # # Convert to true 1-D array from a 2-D, Nx1 array
    # duVec = duVecNby1.reshape((-1,))

    # DM0 = falco.config.Object()

    # Parse the command vector by DM and apply weighting
    dm0 = np.zeros(cvar.NeleAll)
    bounds = np.zeros((cvar.NeleAll, 2))
    if any(mp.dm_ind == 1):
        # print('HERE')
        # print(dm0.shape)
        # print(dm0[cvar.uLegend==1].shape)
        # print(len(mp.dm1.act_ele))
        # print(mp.dm1.V[mp.dm1.act_ele].shape)

        dm1vec = mp.dm1.V.flatten()
        dm1lb = mp.dm1.Vmin - (dm1vec + mp.dm1.biasMap.flatten())
        # dm1lb[dm1lb < -mp.ctrl.ad.dv_max] = -mp.ctrl.ad.dv_max
        dm1ub = mp.dm1.Vmax - (dm1vec + mp.dm1.biasMap.flatten())
        # dm1ub[dm1ub > mp.ctrl.ad.dv_max] = mp.ctrl.ad.dv_max
        dm0[cvar.uLegend == 1] = np.zeros(mp.dm1.Nele)  # dm1vec
        bounds[cvar.uLegend == 1, 0] = dm1lb[mp.dm1.act_ele]
        bounds[cvar.uLegend == 1, 1] = dm1ub[mp.dm1.act_ele]

    if any(mp.dm_ind == 2):
        dm2vec = mp.dm2.V.flatten()
        dm2lb = mp.dm2.Vmin - (dm2vec + mp.dm2.biasMap.flatten())
        # dm2lb[dm2lb < -mp.ctrl.ad.dv_max] = -mp.ctrl.ad.dv_max
        dm2ub = mp.dm2.Vmax - (dm2vec + mp.dm2.biasMap.flatten())
        # dm2ub[dm2ub > mp.ctrl.ad.dv_max] = mp.ctrl.ad.dv_max
        dm0[cvar.uLegend == 2] = np.zeros(mp.dm2.Nele)  # dm2vec
        bounds[cvar.uLegend == 2, 0] = dm2lb[mp.dm2.act_ele]
        bounds[cvar.uLegend == 2, 1] = dm2ub[mp.dm2.act_ele]

    EFendPrev = []
    for iMode in range(mp.jac.Nmode):

        modvar = falco.config.ModelVariables()
        modvar.whichSource = 'star'
        modvar.sbpIndex = mp.jac.sbp_inds[iMode]
        modvar.zernIndex = mp.jac.zern_inds[iMode]
        modvar.starIndex = mp.jac.star_inds[iMode]

        # Calculate E-Field for previous EFC iteration
        EFend = falco.model.compact(mp, modvar, isNorm=True, isEvalMode=False,
                                    useFPM=True, forRevGradModel=False)
        EFendPrev.append(EFend)

    t0 = time.time()
    u_sol = scipy.optimize.minimize(
        falco.model.compact_reverse_gradient, dm0, args=(mp, cvar.Eest, EFendPrev, log10reg),
        method='L-BFGS-B', jac=True, bounds=bounds,
        tol=None, callback=None,
        options={'disp': None, 'ftol': 1e-12, 'gtol': 1e-10, 
                 'maxiter': mp.ctrl.ad.maxiter, 'maxfun': mp.ctrl.ad.maxfun,
                 'maxls': 20, 'iprint': mp.ctrl.ad.iprint},
        )
    t1 = time.time()
    print('Optimization time = %.3f' % (t1-t0))

    duVec = u_sol.x
    print(u_sol.success)
    print(u_sol.nit)

    # Parse the command vector by DM and assign the output commands
    mp, dDM = wrapup(mp, cvar, duVec)

    # Take images and compute average intensity in dark hole
    if mp.ctrl.flagUseModel:  # Get simulated image from compact model
        Itotal = falco.imaging.get_expected_summed_image(mp, cvar, dDM)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
    else:  # Get actual image from full model or testbed
        Itotal = falco.imaging.get_summed_image(mp)
        InormAvg = np.mean(Itotal[mp.Fend.corr.maskBool])
    dDM.Itotal = Itotal

    # Reset voltage commands in mp
    if any(mp.dm_ind == 1):
        mp.dm1.V = DM1V0
    if any(mp.dm_ind == 2):
        mp.dm2.V = DM2V0
    if any(mp.dm_ind == 8):
        mp.dm8.V = DM8V0
    if any(mp.dm_ind == 9):
        mp.dm9.V = DM9V0

    return InormAvg, dDM


def apply_spatial_weighting_to_Jacobian(mp, jacStruct):
    """
    Add spatially-dependent weighting to the control Jacobians.

    Parameters
    ----------
    mp : ModelParameters
        Structure containing optical model parameters
    jacStruct : Object
        Object containing the Jacobians for each DM.

    Returns
    -------
    None.

    """
    for iStar in range(mp.compact.star.count):

        if np.any(mp.dm_ind == 1):
            jacStruct.G1[:, :, mp.jac.star_inds == iStar] = \
                jacStruct.G1[:, :, mp.jac.star_inds == iStar] * \
                np.moveaxis(
                    np.tile(mp.WspatialVec[:, iStar].reshape([-1]),
                            [mp.dm1.Nele, mp.jac.NmodePerStar, 1]), 2, 0)

        if np.any(mp.dm_ind == 2):
            jacStruct.G2[:, :, mp.jac.star_inds == iStar] = \
                jacStruct.G2[:, :, mp.jac.star_inds == iStar] * \
                np.moveaxis(
                    np.tile(mp.WspatialVec[:, iStar].reshape([-1]),
                            [mp.dm2.Nele, mp.jac.NmodePerStar, 1]), 2, 0)

        # if np.any(mp.dm_ind == 8):
        #     jacStruct.G8 = jacStruct.G8*np.moveaxis(
        #         np.tile(mp.WspatialVec[:, None],
        #                 [mp.jac.Nmode, 1, mp.dm8.Nele]), 0, -1)
        # if np.any(mp.dm_ind == 9):
        #     jacStruct.G9 = jacStruct.G9*np.moveaxis(
        #         np.tile(mp.WspatialVec[:, None],
        #                 [mp.jac.Nmode, 1, mp.dm9.Nele]), 0, -1)


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
    if any(mp.dm_ind == 1):
        cvar.DM1Vnom = mp.dm1.V
    if any(mp.dm_ind == 2):
        cvar.DM2Vnom = mp.dm2.V
    if any(mp.dm_ind == 8):
        cvar.DM8Vnom = mp.dm8.V
    if any(mp.dm_ind == 9):
        cvar.DM9Vnom = mp.dm9.V

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
    mp.ctrl.uLegend = cvar.uLegend

    return None


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
    if any(mp.dm_ind == 1):
        dDM1Vvec = np.zeros(mp.dm1.NactTotal)
        dDM1Vvec[mp.dm1.act_ele] = mp.dm1.weight*duVec[cvar.uLegend == 1]
        dDM.dDM1V = dDM1Vvec.reshape((mp.dm1.Nact, mp.dm1.Nact))
    if any(mp.dm_ind == 2):
        dDM2Vvec = np.zeros(mp.dm2.NactTotal)
        dDM2Vvec[mp.dm2.act_ele] = mp.dm2.weight*duVec[cvar.uLegend == 2]
        dDM.dDM2V = dDM2Vvec.reshape((mp.dm2.Nact, mp.dm2.Nact))
    if any(mp.dm_ind == 8):
        dDM.dDM8V = np.zeros(mp.dm8.NactTotal)
        dDM.dDM8V[mp.dm8.act_ele] = mp.dm8.weight*duVec[cvar.uLegend == 8]
    if any(mp.dm_ind == 9):
        dDM.dDM9V = np.zeros(mp.dm9.NactTotal)
        dDM.dDM9V[mp.dm9.act_ele] = mp.dm9.weight*duVec[cvar.uLegend == 9]

    # Enforce tied actuator pair commands.
    #   Assign command of first actuator to the second as well.
#    if any(mp.dm_ind == 8):
#        for ti=1:size(mp.dm8.tied,1)
#            dDM.dDM8V(mp.dm8.tied(ti,2)) = dDM.dDM8V(mp.dm8.tied(ti,1));
#
#    if any(mp.dm_ind == 9):
#        for ti=1:size(mp.dm9.tied,1)
#            dDM.dDM9V(mp.dm9.tied(ti,2)) = dDM.dDM9V(mp.dm9.tied(ti,1));

    # Combine the delta command with the previous command
    if any(mp.dm_ind == 1):
        mp.dm1.V = cvar.DM1Vnom + dDM.dDM1V
    if any(mp.dm_ind == 2):
        mp.dm2.V = cvar.DM2Vnom + dDM.dDM2V
    if any(mp.dm_ind == 8):
        mp.dm8.V = cvar.DM8Vnom + dDM.dDM8V
    if any(mp.dm_ind == 9):
        mp.dm9.V = cvar.DM9Vnom + dDM.dDM9V

    return mp, dDM


def set_utu_scale_fac(mp):
    falco.imaging.calc_psf_norm_factor(mp)

    # Save the original values and then use only the specified subset of actuators
    if np.any(mp.dm_ind == 1):
        dm1_act_ele = copy.copy(mp.dm1.act_ele)
        dm1_Nele = copy.copy(mp.dm1.Nele)
        mp.dm1.act_ele = np.arange(mp.dm1.NactTotal, dtype=int)[mp.ctrl.ad.dm1_act_mask_for_jac_norm]
        mp.dm1.Nele = len(mp.dm1.act_ele)

    if np.any(mp.dm_ind == 2):
        dm2_act_ele = copy.copy(mp.dm2.act_ele)
        dm2_Nele = copy.copy(mp.dm2.Nele)
        mp.dm2.act_ele = np.arange(mp.dm2.NactTotal, dtype=int)[mp.ctrl.ad.dm2_act_mask_for_jac_norm]
        mp.dm2.Nele = len(mp.dm2.act_ele)

    # Compute the Jacobian for some actuators
    jacStruct = falco.model.jacobian(mp)

    print('Using the Jacobian to make other matrices...', end='')

    # Compute matrices for linear control with regular EFC
    cvar = falco.config.Object()
    # Make the vector of total DM commands from before
    u1 = mp.dm1.V.reshape(mp.dm1.NactTotal)[mp.dm1.act_ele] if any(mp.dm_ind == 1) else np.array([])
    u2 = mp.dm2.V.reshape(mp.dm2.NactTotal)[mp.dm2.act_ele] if any(mp.dm_ind == 2) else np.array([])
    cvar.uVec = np.concatenate((u1, u2))
    cvar.NeleAll = cvar.uVec.size

    cvar.GstarG_wsum = np.zeros((cvar.NeleAll, cvar.NeleAll))
    cvar.RealGstarEab_wsum = np.zeros((cvar.NeleAll, 1))

    for iMode in range(mp.jac.Nmode):

        Gstack = np.zeros((mp.Fend.corr.Npix, 1), dtype=complex)  # Initialize a row to concatenate onto
        if any(mp.dm_ind == 1):
            Gstack = np.hstack((Gstack, np.squeeze(jacStruct.G1[:, :, iMode])))
        if any(mp.dm_ind == 2):
            Gstack = np.hstack((Gstack, np.squeeze(jacStruct.G2[:, :, iMode])))
        Gstack = Gstack[:, 1:]  # Remove the column used for initialization

        # Square matrix part stays the same if no re-linearization has occurrred.
        cvar.GstarG_wsum += mp.jac.weights[iMode]*np.real(np.conj(Gstack).T @ Gstack)

    # Make the regularization matrix. (Define only diagonal here to save RAM.)
    cvar.EyeGstarGdiag = np.max(np.diag(cvar.GstarG_wsum))*np.ones(cvar.NeleAll)
    cvar.EyeNorm = np.max(np.diag(cvar.GstarG_wsum))
    print('done.')

    # Reset back to all actuators
    if np.any(mp.dm_ind == 1):
        mp.dm1.act_ele = dm1_act_ele  # list(range(mp.dm1.NactTotal))
        mp.dm1.Nele = dm1_Nele  # len(mp.dm1.act_ele)
    if np.any(mp.dm_ind == 2):
        mp.dm2.act_ele = dm2_act_ele  # list(range(mp.dm2.NactTotal))
        mp.dm2.Nele = dm2_Nele  # len(mp.dm2.act_ele)

    # Set the scaling factor in the cost function for the squared actuator voltage term
    mp.ctrl.ad.utu_scale_fac = cvar.EyeNorm

    print('mp.ctrl.ad.utu_scale_fac = %.4g' % mp.ctrl.ad.utu_scale_fac)
