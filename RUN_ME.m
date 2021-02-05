% Turbo coded QPSK over AWGN - logMAP

close all
clear all
clc
%---------------- SIMULATION PARAMETERS ------------------------------------
SNR_dB = 3; % SNR per bit in dB (in logarithmic scale)
sim_runs = 1*(10^1); % simulation runs
frame_size = 1024; % frame size
num_bit = 0.5*frame_size; % number of data bits (overall rate is 1/2)
SNR = 10^(0.1*SNR_dB); % SNR per bit in linear scale
noise_var_1D = 2*2/(2*SNR); % 1D noise variance
%--------------------------------------------------------------------------
%    Generator polynomial of the component encoders
gen_poly = ldiv2([1 0 1],[1 1 1],num_bit); % using long division method

%  Interleaver and deinterleaver mapping of the turbo code 
intr_map = randperm(num_bit);
deintr_map = deintrlv((1:num_bit),intr_map);

%--------------------------------------------------------------------------
C_Ber = 0; % channel erros
tic()
%--------------------------------------------------------------------------
for frame_cnt = 1:sim_runs
%                           TRANSMITTER
%Source
a = randi([0 1],1,num_bit); % data

% Turbo encoder
% component encoder 1
b1 = zeros(1,2*num_bit); % encoder 1 output initialization
b1(1:2:end) = a; % systematic bit
temp1 = mod(conv(gen_poly,a),2); % linear convolution with the generator polynomial
b1(2:2:end) = temp1(1:num_bit); % parity bit
% component encoder 2
b2 = zeros(1,2*num_bit); % encoder 2 output initialization
b2(1:2:end) = a(intr_map); % systematic bit
temp2 = mod(conv(gen_poly,b2(1:2:end)),2); % linear convolution with the generator polynomial
b2(2:2:end) = temp2(1:num_bit); % parity bit

% QPSK mapping (according to the set partitioning principles)
mod_sig1 = 1-2*b1(1:2:end) + 1i*(1-2*b1(2:2:end));
mod_sig2 = 1-2*b2(1:2:end) + 1i*(1-2*b2(2:2:end));
mod_sig = [mod_sig1 mod_sig2];

%--------------------------------------------------------------------------
%                            CHANNEL   
% AWGN
white_noise = sqrt(noise_var_1D)*randn(1,frame_size)+1i*sqrt(noise_var_1D)*randn(1,frame_size); 
Chan_Op = mod_sig + white_noise; % Chan_Op stands for channel output
%--------------------------------------------------------------------------
%                          RECEIVER 

% Branch metrices for the BCJR
QPSK_SYM = zeros(4,frame_size);
QPSK_SYM(1,:) = (1+1i)*ones(1,frame_size);
QPSK_SYM(2,:) = (1-1i)*ones(1,frame_size);
QPSK_SYM(3,:) = (-1+1i)*ones(1,frame_size);
QPSK_SYM(4,:) = (-1-1i)*ones(1,frame_size);

Dist = zeros(4,frame_size);
 Dist(1,:)=abs(Chan_Op-QPSK_SYM(1,:)).^2;
 Dist(2,:)=abs(Chan_Op-QPSK_SYM(2,:)).^2;
 Dist(3,:)=abs(Chan_Op-QPSK_SYM(3,:)).^2;
 Dist(4,:)=abs(Chan_Op-QPSK_SYM(4,:)).^2;
 log_gamma = -Dist/(2*noise_var_1D);
 
log_gamma1 = log_gamma(:,1:num_bit); % branch metrices for component decoder 1
log_gamma2 = log_gamma(:,num_bit+1:end); % branch metrices for component decoder 2
 
% a priori LLR for component decoder 1 for 1st iteration
LLR = zeros(1,num_bit);

% iterative logMAP decoding
LLR = log_BCJR(LLR,log_gamma1,num_bit); % outputs extrinsic information
LLR = log_BCJR(LLR(intr_map),log_gamma2,num_bit); %1

LLR = log_BCJR(LLR(deintr_map),log_gamma1,num_bit);
LLR = log_BCJR(LLR(intr_map),log_gamma2,num_bit); %2

LLR = log_BCJR(LLR(deintr_map),log_gamma1,num_bit);
LLR = log_BCJR(LLR(intr_map),log_gamma2,num_bit); %3

LLR = log_BCJR(LLR(deintr_map),log_gamma1,num_bit);
LLR = log_BCJR(LLR(intr_map),log_gamma2,num_bit); %4

LLR = log_BCJR(LLR(deintr_map),log_gamma1,num_bit);
LLR = log_BCJR(LLR(intr_map),log_gamma2,num_bit); %5

LLR = log_BCJR(LLR(deintr_map),log_gamma1,num_bit);
LLR = log_BCJR(LLR(intr_map),log_gamma2,num_bit); %6

LLR = log_BCJR(LLR(deintr_map),log_gamma1,num_bit);
LLR = log_BCJR(LLR(intr_map),log_gamma2,num_bit); %7

LLR = log_BCJR(LLR(deintr_map),log_gamma1,num_bit);
LLR = log_BCJR_END(LLR(intr_map),log_gamma2,num_bit); % 8: outputs aposteriori LLRs

% hard decision 
LLR = LLR(deintr_map);
dec_data = LLR<0;

 % Calculating total bit errors
C_Ber = C_Ber + nnz(dec_data-a); 
end

BER = C_Ber/(sim_runs*num_bit)
toc()

