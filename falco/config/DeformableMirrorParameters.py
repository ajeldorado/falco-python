import numpy as np
import os
import scipy.io
import falco.utils
from falco.utils import _spec_arg

_influence_function_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "influence_dm5v2.mat")


class DeformableMirrorParameters:
    class _base_dm:
        def __init__(self, **kwargs):
            self.dm_spacing = _spec_arg("dm_spacing", kwargs, 0.001)
            self.dx_inf0 = _spec_arg("dx_inf0", kwargs, 0.0001)
            self.xc = _spec_arg("xc", kwargs, 23.5)
            self.dummy = _spec_arg("dummy", kwargs, 1)
            self.ytilt = _spec_arg("ytilt", kwargs, 0)
            self.xtilt = _spec_arg("xtilt", kwargs, 0)
            self.Nact = _spec_arg("Nact", kwargs, 48)
            self.edgeBuffer = _spec_arg("edgeBuffer", kwargs, 1)
            self.maxAbsV = _spec_arg("maxAbsV", kwargs, 125)
            self.yc = _spec_arg("yc", kwargs, 23.5)
            self.zrot = _spec_arg("zrot", kwargs, 0)
            self.VtoH = _spec_arg("VtoH", kwargs, 1e-9*np.ones((1, self.Nact)))
            self.inf0 = _spec_arg("inf0", kwargs, scipy.io.loadmat(
                _influence_function_file)["inf0"])

    def __init__(self, **kwargs):
        self.dm_weights = _spec_arg("dm_weights", kwargs, [1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.dm_ind = _spec_arg("dm_ind", kwargs, [1, 2])
        self.maxAbsdV = _spec_arg("maxAbsdV", kwargs, 30)
        self.dm1 = _spec_arg("dm1", kwargs, self._base_dm())
        self.dm2 = _spec_arg("dm2", kwargs, self._base_dm())

    def falco_gen_dm_poke_cube(self, mp, dx_dm, flagGenCube=True):
        #Compute sampling of the pupil. Assume that it is square.
        dm.dx_dm = dx_dm
        dm.dx = dx_dm

        # Compute coordinates of original influence function
        Ninf0 = len(dm.inf0) #Number of points across the influence function at its native resolution
        #x_inf0 = (-(Ninf0-1)/2:(Ninf0-1)/2)*dm.dx_inf0 # True for even- or odd-sized influence function maps as long as they are centered on the array.
        x_inf0 = np.arange(-(Ninf0-1)//2, (Ninf0-1)//2)*dm.dx_inf0

        Ndm0 = falco.utils.ceil_even( Ninf0 + (dm.Nact - 1)*(dm.dm_spacing/dm.dx_inf0) ) #Number of points across the DM surface at native influence function resolution
        dm.NdmMin = ceil_even( Ndm0*(dm.dx_inf0/dm.dx))+2 #Number of points across the (un-rotated) DM surface at new, desired resolution.
        dm.Ndm = ceil_even( max(np.abs([np.sqrt(2)*np.cos(np.radians(45-dm.zrot)), np.sqrt(2)*np.sind(np.radians(45-dm.zrot))]))*Ndm0*(dm.dx_inf0/dm.dx))+2 #Number of points across the array to fully contain the DM surface at new, desired resolution and z-rotation angle.

        Xinf0,Yinf0 = np.meshgrid(x_inf0)
        """   
%--Compute list of initial actuator center coordinates (in actutor widths).
if(dm.flag_hex_array) %--Hexagonal, hex-packed grid
    Nrings = dm.Nrings;
    x_vec = [];
    y_vec = [];
    % row number (rowNum) is 1 for the center row and 2 is above it, etc.
    % Nacross is the total number of segments across that row
    for rowNum = 1:Nrings
        Nacross = 2*Nrings - rowNum; % Number of actuators across at that row (for hex tiling in a hex shape)
        yval = sqrt(3)/2*(rowNum-1);
        bx = Nrings - (rowNum+1)/2; % x offset from origin

        xs = (0:Nacross-1).' - bx; % x values are 1 apart
        ys = yval*ones(Nacross,1); % same y-value for the entire row

        if(rowNum==1)
            x_vec = [x_vec;xs];
            y_vec = [y_vec;ys]; 
        else
            x_vec = [x_vec;xs;xs];
            y_vec = [y_vec;ys;-ys]; % rows +/-n have +/- y coordinates
        end
    end    
else %--Square grid
    [dm.Xact,dm.Yact] = meshgrid((0:dm.Nact-1)-dm.xc,(0:dm.Nact-1)-dm.yc); % in actuator widths
    x_vec = dm.Xact(:);
    y_vec = dm.Yact(:);
end
dm.NactTotal = length(x_vec); %--Total number of actuators in the 2-D array



tlt  = zeros(1, 3);
tlt(1) = dm.xtilt;
tlt(2) = dm.ytilt;
tlt(3) = -dm.zrot;

sa   = sind(tlt(1));
ca   = cosd(tlt(1));
sb   = sind(tlt(2));
cb   = cosd(tlt(2));
sg   = sind(tlt(3));
cg   = cosd(tlt(3));

if zyx == true
    Mrot = [               cb * cg,               -cb * sg,       sb, 0.0; ...
            ca * sg + sa * sb * cg, ca * cg - sa * sb * sg, -sa * cb, 0.0; ...
            sa * sg - ca * sb * cg, sa * cg + ca * sb * sg,  ca * cb, 0.0; ...
                               0.0,                    0.0,      0.0, 1.0];
else
    Mrot = [ cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, 0.0; ...
             cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg, 0.0; ...
            -sb,      sa * cb,                ca * cb,                0.0; ...
                 0.0,                    0.0,                    0.0, 1.0];
end

for iact=1:dm.NactTotal
    xyzVals = [x_vec(iact); y_vec(iact); 0; 1];
    xyzValsRot = Mrot*xyzVals;
    dm.xy_cent_act(:,iact) = xyzValsRot(1:2);
end


  
N0 = max(size(dm.inf0));
Npad = ceil_odd( sqrt(2)*max(size(dm.inf0)) );
inf0pad = zeros(Npad,Npad);

inf0pad( ceil(Npad/2)-floor(N0/2):ceil(Npad/2)+floor(N0/2), ceil(Npad/2)-floor(N0/2):ceil(Npad/2)+floor(N0/2) ) = dm.inf0;

[ydim,xdim] = size(inf0pad);

xd2  = fix(xdim / 2) + 1;
yd2  = fix(ydim / 2) + 1;
cx   = ([1 : xdim] - xd2) ;
cy   = ([1 : ydim] - yd2) ;
[Xs0, Ys0] = meshgrid(cx, cy);

xsNew = 0*Xs0;
ysNew = 0*Ys0;
 
for ii=1:numel(Xs0)
    xyzVals = [Xs0(ii); Ys0(ii); 0; 1];
    xyzValsRot = Mrot*xyzVals;
    xsNew(ii) = xyzValsRot(1);
    ysNew(ii) = xyzValsRot(2);
end

% Calculate the interpolated DM grid (set extrapolated values to 0.0)
dm.infMaster = griddata(xsNew,ysNew,inf0pad,Xs0,Ys0,'cubic');%,'cubic',0);
dm.infMaster(isnan(dm.infMaster)) = 0;
 
% x_inf0 = (-(Npad-1)/2:(Npad-1)/2)*dm.dx_inf0; % True for even- or odd-sized influence function maps as long as they are centered on the array.
% [Xinf0,Yinf0] = meshgrid(x_inf0);
 



%--Crop down the influence function until it has no zero padding left
infSum = sum(dm.infMaster(:));
infDiff = 0; counter = 0;
while( abs(infDiff) <= 1e-7)
    counter = counter + 2;
    Ninf0pad = length(dm.infMaster)-counter; %--Number of points across the rotated, cropped-down influence function at the original resolution
    infDiff = infSum - sum(sum( dm.infMaster(1+counter/2:end-counter/2,1+counter/2:end-counter/2) )); %--Subtract an extra 2 to negate the extra step that overshoots.
end
counter = counter - 2;
Ninf0pad = length(dm.infMaster)-counter; %Ninf0pad = Ninf0pad+2;
infMaster2 = dm.infMaster(1+counter/2:end-counter/2,1+counter/2:end-counter/2); % padOrCropEven(dm.infMaster,Ncrop); %--The cropped-down Lyot stop for the compact model       
% figure; imagesc(log10(abs(infMaster2)));


dm.infMaster = infMaster2;
Npad = Ninf0pad;

x_inf0 = (-(Npad-1)/2:(Npad-1)/2)*dm.dx_inf0; % True for even- or odd-sized influence function maps as long as they are centered on the array.
[Xinf0,Yinf0] = meshgrid(x_inf0);



%%%%%%%%%%%%%%%%%%%%%%%


% %--Apply x- and y-projections and then z-rotation to the original influence
% %    function to make a master influence function.
% dm.Xrot = Xinf0/cosd(dm.ytilt); % Divide coords by projection factor to squeeze the influence function
% dm.Yrot = Yinf0/cosd(dm.xtilt);
% dm.infMaster = interp2(Xinf0,Yinf0,dm.inf0,dm.Xrot,dm.Yrot,'cubic',0);
% dm.infMaster = imrotate(dm.infMaster,dm.zrot,'bicubic','crop');

%--Compute the size of the postage stamps.
Nbox = ceil_even(((Ninf0pad*dm.dx_inf0)/dx_dm)); % Number of points across the influence function in the pupil file's spacing. Want as even
dm.Nbox = Nbox;
%--Also compute their padded sizes for the angular spectrum (AS) propagation between P2 and DM1 or between DM1 and DM2
Nmin = ceil_even( max(mp.sbp_center_vec)*max(abs([mp.d_P2_dm1, mp.d_dm1_dm2,(mp.d_P2_dm1+mp.d_dm1_dm2)]))/dx_dm^2 ); %--Minimum number of points across for accurate angular spectrum propagation
% dm.NboxAS = 2^(nextpow2(max([Nbox,Nmin])));  %--Zero-pad for FFTs in A.S. propagation. Uses a larger array if the max sampling criterion for angular spectrum propagation is violated
dm.NboxAS = max([Nbox,Nmin]);  %--Uses a larger array if the max sampling criterion for angular spectrum propagation is violated

% dm.NdmPad = ceil_even( dm.Ndm + (dm.NboxAS-dm.Nbox) ); %--Number of points across the DM surface (with padding for angular spectrum propagation) at new, desired resolution.

% if( Nbox < Nmin ) %--Use a larger array if the max sampling criterion for angular spectrum propagation is violated
%     dm.NboxAS = 2^(nextpow2(Nmin)); %2*ceil(1/2*min(mp.sbp_center_vec)*mp.d_dm1_dm2/dx_dm^2);
% else
%     dm.NboxAS = 2^(nextpow2(Nbox));
% end

%% Pad the pupil to at least the size of the DM(s) surface(s) to allow all actuators to be located outside the pupil.
% (Same for both DMs)

%--Find actuator farthest from center:
dm.r_cent_act = sqrt(dm.xy_cent_act(1,:).^2 + dm.xy_cent_act(2,:).^2);
dm.rmax = max(abs(dm.r_cent_act));
NpixPerAct = dm.dm_spacing/dx_dm;
if(dm.flag_hex_array)
    %dm.NdmPad = 2*ceil(1/2*Nbox*2) + 2*ceil((1/2*2*(dm.rmax)*dm.dx_inf0_act)*Nbox); %2*ceil((dm.rmax+3)*dm.dm_spacing/Dpup*Npup);
    dm.NdmPad = ceil_even((2*(dm.rmax+2))*NpixPerAct + 1); % padded 2 actuators past the last actuator center to avoid trying to index outside the array 
else
%         dm.NdmPad = ceil_even( sqrt(2)*( 2*(dm.rmax*NpixPerAct + 1)) ); % padded 1/2 an actuator past the farthest actuator center (on each side) to prevent indexing outside the array 
    dm.NdmPad = ceil_even( ( dm.NboxAS + 2*(1+ (max(max(abs(dm.xy_cent_act)))+0.5)*NpixPerAct)) ); % DM surface array padded by the width of the padded influence function to prevent indexing outside the array. The 1/2 term is because the farthest actuator center is still half an actuator away from the nominal array edge. 
end

%--Compute coordinates (in meters) of the full DM array
if(strcmpi(dm.centering,'pixel')  ) 
    dm.x_pupPad = (-dm.NdmPad/2:(dm.NdmPad/2 - 1))*dx_dm; % meters, coords for the full DM arrays. Origin is centered on a pixel
else
    dm.x_pupPad = (-(dm.NdmPad-1)/2:(dm.NdmPad-1)/2)*dx_dm; % meters, coords for the full DM arrays. Origin is centered between pixels for an even-sized array
end
dm.y_pupPad = dm.x_pupPad;



%% DM: (use NboxPad-sized postage stamps)

if(flagGenCube)
    if(dm.flag_hex_array==false)
        fprintf('  Influence function padded from %d to %d points for A.S. propagation.\n',Nbox,dm.NboxAS);
%         fprintf('  Influence function padded to 2^nextpow2(%d) = %d for A.S. propagation.\n',2*ceil(1/2*max([Nbox,Nmin])),dm.NboxAS);
    end
    tic
    fprintf('Computing datacube of DM influence functions... ');

    %--Find the locations of the postage stamps arrays in the larger pupilPad array
    dm.xy_cent_act_inPix = dm.xy_cent_act*(dm.dm_spacing/dx_dm); % Convert units to pupil-file pixels
%     if(strcmpi(dm.centering,'pixel')  ) 
       dm.xy_cent_act_inPix = dm.xy_cent_act_inPix + 0.5; %--For the half-pixel offset if pixel centered. 
%     end
    dm.xy_cent_act_box = round(dm.xy_cent_act_inPix); % Center locations of the postage stamps (in between pixels), in actuator widths
    dm.xy_cent_act_box_inM = dm.xy_cent_act_box*dx_dm; % now in meters 
    dm.xy_box_lowerLeft = dm.xy_cent_act_box + (dm.NdmPad-Nbox)/2 + 1; % indices of pixel in lower left of the postage stamp within the whole pupilPad array

    %--Starting coordinates (in actuator widths) for updated influence function. This is
    % interpixel centered, so do not translate!
    dm.x_box0 = (-(Nbox-1)/2:(Nbox-1)/2)*dx_dm;
    [dm.Xbox0,dm.Ybox0] = meshgrid(dm.x_box0); %--meters, interpixel-centered coordinates for the master influence function

    %--Limit the actuators used to those within 1 actuator width of the pupil
    r_cent_act_box_inM = sqrt(dm.xy_cent_act_box_inM(1,:).^2 + dm.xy_cent_act_box_inM(2,:).^2);
    %--Compute and store all the influence functions:
    dm.inf_datacube = zeros(Nbox,Nbox,dm.NactTotal);%dm.Nact^2); %--initialize array of influence function "postage stamps"
    dm.act_ele = []; % Indices of nonzero-ed actuators
    for iact=1:dm.NactTotal %dm.Nact^2
%         if(r_cent_act_box_inM(iact) < D/2 + dm.edgeBuffer*Nbox*dx_dm) %--Don't use actuators too far outside the beam
            dm.act_ele = [dm.act_ele; iact]; % Add actuator index to the keeper list
            dm.Xbox = dm.Xbox0 - (dm.xy_cent_act_inPix(1,iact)-dm.xy_cent_act_box(1,iact))*dx_dm; % X = X0 -(x_true_center-x_box_center)
            dm.Ybox = dm.Ybox0 - (dm.xy_cent_act_inPix(2,iact)-dm.xy_cent_act_box(2,iact))*dx_dm; % Y = Y0 -(y_true_center-y_box_center)
            dm.inf_datacube(:,:,iact) = interp2(Xinf0,Yinf0,dm.infMaster,dm.Xbox,dm.Ybox,'spline',0);
%         end
    end
    
    fprintf('done.  Time = %.1fs\n',toc);

else
    dm.act_ele = (1:dm.NactTotal).';    
end

end %--END OF FUNCTION

        """
    def init_ws(self, mp):
        self.flag_hex_array = False
        self.flagZYX = False

        self.dm1.NactTotal=0
        self.dm2.NactTotal=0

        self.dm1.centering = mp.centering;
        self.dm2.centering = mp.centering;

        if any(self.dm_ind):
            self.dm1.compact = self.dm1
            #self.dm1 = falco_gen_dm_poke_cube(self.dm1, mp, mp.P2.full.dx,'NOCUBE')
            #self.dm1.compact = falco_gen_dm_poke_cube(self.dm1.compact, mp, mp.P2.compact.dx);

        """
if( any(DM.dm_ind==2) ) %if(isfield(DM.dm2,'inf_datacube')==0 && any(DM.dm_ind==2) )
    DM.dm2.compact = DM.dm2;
    DM.dm2.dx = mp.P2.full.dx;
    DM.dm2.compact.dx = mp.P2.compact.dx;
    
    DM.dm2 = falco_gen_dm_poke_cube(DM.dm2, mp, mp.P2.full.dx, 'NOCUBE');
    DM.dm2.compact = falco_gen_dm_poke_cube(DM.dm2.compact, mp, mp.P2.compact.dx);
%     DM.dm2 = falco_gen_dm_poke_cube_PROPER(DM.dm2,mp,'NOCUBE');
%     DM.dm2.compact = falco_gen_dm_poke_cube_PROPER(DM.dm2.compact,mp);
end



% % DM.dm1.Ndm = min([ceil_even(mp.P1.full.Narr*(DM.dm1.dm_spacing*DM.dm1.Nact)/(mp.Ddm1)), DM.dm1.NdmPad]); %--Number of points across to crop the surface to for storage
% % DM.dm2.Ndm = min([ceil_even(mp.P1.full.Narr*(DM.dm2.dm_spacing*DM.dm2.Nact)/(mp.Ddm1)), DM.dm2.NdmPad]); %--Number of points across to crop the surface to for storage
% % DM.dm1.compact.Ndm = min([ceil_even(mp.P1.compact.Narr*(DM.dm1.compact.dm_spacing*DM.dm1.compact.Nact)/(mp.Ddm1)), DM.dm1.compact.NdmPad]); %--Number of points across to crop the surface to for storage
% % DM.dm2.compact.Ndm = min([ceil_even(mp.P1.compact.Narr*(DM.dm2.compact.dm_spacing*DM.dm1.compact.Nact)/(mp.Ddm1)), DM.dm2.compact.NdmPad]); %--Number of points across to crop the surface to for storage

%--Initial DM voltages
DM.dm1.V = zeros(DM.dm1.Nact);
DM.dm2.V = zeros(DM.dm2.Nact);

%--Peak-to-Valley DM voltages
DM.dm1.Vpv = zeros(mp.Nitr,1);
DM.dm2.Vpv = zeros(mp.Nitr,1);

%--First delta DM settings are zero (for covariance calculation in Kalman filters or robust controllers)
DM.dm1.dV = zeros(DM.dm1.Nact,DM.dm1.Nact);  % delta voltage on DM1;
DM.dm2.dV = zeros(DM.dm2.Nact,DM.dm2.Nact);  % delta voltage on DM2;

%--Store the DM commands 
DM.dm1.Vall = zeros(DM.dm1.Nact,DM.dm1.Nact,mp.Nitr+1);
DM.dm2.Vall = zeros(DM.dm2.Nact,DM.dm2.Nact,mp.Nitr+1);

%--Initialize the number of actuators used.
DM.dm1.Nele=[]; DM.dm2.Nele=[];  DM.dm3.Nele=[];  DM.dm4.Nele=[];  DM.dm5.Nele=[];  DM.dm6.Nele=[];  DM.dm7.Nele=[];  DM.dm8.Nele=[];  DM.dm9.Nele=[]; %--Initialize for Jacobian calculations later. 
if(any(DM.dm_ind==1)); DM.dm1.Nele = length(DM.dm1.act_ele); end
if(any(DM.dm_ind==2)); DM.dm2.Nele = length(DM.dm2.act_ele); end
if(any(DM.dm_ind==9)); DM.dm9.Nele = length(DM.dm9.act_ele); end
DM.NelePerDMvec = [length(DM.dm1.Nele), length(DM.dm2.Nele), length(DM.dm3.Nele), length(DM.dm4.Nele), length(DM.dm5.Nele), length(DM.dm6.Nele), length(DM.dm7.Nele), length(DM.dm8.Nele), length(DM.dm9.Nele) ];
DM.NactTotals = [DM.dm1.NactTotal, DM.dm2.NactTotal, DM.dm3.NactTotal, DM.dm4.NactTotal, DM.dm5.NactTotal, DM.dm6.NactTotal, DM.dm7.NactTotal, DM.dm8.NactTotal, DM.dm9.NactTotal]; 
DM.NactTotal = sum(DM.NactTotals);

%% Array Sizes for Angular Spectrum Propagation with FFTs

%--Compact Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
if( any(DM.dm_ind==1) && any(DM.dm_ind==2) )
    NdmPad = 2.^ceil(1 + log2(max([DM.dm1.compact.NdmPad,DM.dm2.compact.NdmPad]))); 
elseif(  any(DM.dm_ind==1) )
    NdmPad = 2.^ceil(1 + log2(DM.dm1.compact.NdmPad));
elseif(  any(DM.dm_ind==2) )
    NdmPad = 2.^ceil(1 + log2(DM.dm2.compact.NdmPad));
end
while( (NdmPad < min(mp.sbp_center_vec)*abs(mp.d_dm1_dm2)/mp.P2.full.dx^2) || (NdmPad < min(mp.sbp_center_vec)*abs(mp.d_P2_dm1)/mp.P2.compact.dx^2) ) %--Double the zero-padding until the angular spectrum sampling requirement is not violated
    NdmPad = 2*NdmPad; 
end
DM.compact.NdmPad = NdmPad;

%--Full Model: Set nominal DM plane array sizes as a power of 2 for angular spectrum propagation with FFTs
if( any(DM.dm_ind==1) && any(DM.dm_ind==2) )
    NdmPad = 2.^ceil(1 + log2(max([DM.dm1.NdmPad,DM.dm1.NdmPad]))); 
elseif(  any(DM.dm_ind==1) )
    NdmPad = 2.^ceil(1 + log2(DM.dm1.NdmPad));
elseif(  any(DM.dm_ind==2) )
    NdmPad = 2.^ceil(1 + log2(DM.dm2.NdmPad));
end
while( (NdmPad < min(mp.lam_array)*abs(mp.d_dm1_dm2)/mp.P2.full.dx^2) || (NdmPad < min(mp.lam_array)*abs(mp.d_P2_dm1)/mp.P2.full.dx^2) ) %--Double the zero-padding until the angular spectrum sampling requirement is not violated
    NdmPad = 2*NdmPad; 
end
DM.full.NdmPad = NdmPad;
        """
