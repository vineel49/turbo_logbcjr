% Trellis for the Convolutional encoder with generator matrix given by
% [1 (1+D^2)/(1+D+D^2)]
function [P_State,P_Ip,Ga_Inx,N_State,Gb_Inx]= Get_Trellis()
Ga_Inx = [1,4; 2,3; 1,4; 2,3];  % Gamma indices for alpha recursion
P_State = [1,2; 4,3; 2,1; 3,4]; % previous state
P_Ip =  [1,2;1,2;1,2;1,2] ; % Previous input.
N_State = [1,3; 3,1; 4,2; 2,4]; % Next state
Gb_Inx = [1,4; 1,4; 2,3; 2,3]; % Gamma indices for beta recursion
end
