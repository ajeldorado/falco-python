import falco
import falco.configs
import falco.mask

mp = falco.config.ModelParameters()
mp.init_ws()
print(mp.P2.full.dx)
#falco.configs.falco_config_gen_chosen_pupil(mp)
falco.mask.falco_gen_pupil_WFIRST_CGI_180718(mp.P1.full.Nbeam, mp.centering)
print(mp.P2.full.dx)

