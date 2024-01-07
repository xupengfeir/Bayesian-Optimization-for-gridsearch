% -------------------------------------------------------------------------
% Estimate an ARX discrete-time model from I/O data 
% Copyright@Jiandong Wang

% close all; clear all;
% disp('--------------------------------------------------');
% Ts = 1; % sampling period
% 
% %% True System
% num = 1.25; 
% den = [40 1];
% theta = 2; % delay ;
% 
% %% PID parameters 
% Kc = 0.7; 
% Ti = 20; 
% Td = 0; 
% alpha = 10; %Anti-derivative kick
%    
% %% Process noise configuration
% nvar =1*0.001; 
% n_num = 1;
% n_den = 1; 
% % n_den = [1 -0.9];  % low-pass spectrum noise 
% % n_den = [1 0.5];  % high-pass spectrum noise
% 
% %% Define the excitation signal
% r0 = 1*10+ 1*[zeros(1, 900), zeros(1, 130), zeros(1,30), ones(1, 1000), ones(1,40), ones(1, 40), ones(1,40), ones(1, 9000)]';  
% r0_L = length(r0); r0t = linspace(0, Ts*(r0_L-1), r0_L)'; r = [r0t, r0]; 
% 
% %% ClosedLoopSimulations
% sim('ClosedLoopStepResp_FBD');
% 
% %% Prepare the data generated from the simulink model
% u = u0.signals.values;  % controller output
% y = y0.signals.values;  % process output
% r = ref0.signals.values;  % reference
% t = y0.time;
% 
% figure(1), subplot(2,1,1), plot(t, y, 'linewidth', 1), hold on; plot(t, r,'k:', 'linewidth', 2 ); xlabel('time (sec)'), 
% title('y & r');
% figure(1), subplot(2,1,2), plot(t, u, 'o', 'linewidth', 1), xlabel('time (sec)'), title('u');
% 
% %% Select the data for model ID and validation
% datae_ts = 1000; 
% datae_te = 1400; 
% 
% uSel = u([datae_ts:datae_te]); 
% ySel = y([datae_ts:datae_te]); 
% rSel = r0([datae_ts:datae_te]); 
% tSel = t([datae_ts:datae_te]); 
% 
% figure(2), subplot(2,1,1), plot(tSel, ySel, 'linewidth', 1), 
% hold on; plot(tSel, rSel,'k:', 'linewidth', 2 ); xlabel('time (sec)'), title('y & r'); 
% subplot(2,1,2), plot(tSel, uSel, 'linewidth', 1), xlabel('time (sec)'), title('u');
% 
% %% Estimate the ARX model 
% yID = ySel-10; 
% uID = uSel-4;
% 
% tID = tSel; 
% 
% Vfit = [];  
% Vfit_loss = [];
% 
% 
% yhat_fit = modelID_ARX(yID, uID, Na, Nb, Ts);

% 目标函数，只保留Yhat_fit作为目标函数的输出，Yhat_fit应取最大值
function yhat_fit = ModelID_ARX(yID, uID, Na, Nb, Ts)
    yhat_fit = [];
    for i=1:length(Na) 
        na = Na(i);nb = Nb(i);
        N = length(yID); 
        
        % define the index for the first sample in the regressor of Y
        if na>(nb+1)
            t0 = na; 
        else
            t0 = nb + 1;
        end
        
        % form the regressor matrix 
        Y = yID(t0+1:N);  % output vector [y(t0), y(t0-1), ..., y(N)]^T
        Yphi = []; 
        for i = 1: na
            Yphi = [Yphi, -yID((t0+1-i):(N-i))];  % regressor for output: [y(t0-i), y(t0-i-1), ..., y(N-i)]^T
        end
        Uphi = []; 
        for i = 1: nb 
            Uphi = [Uphi, uID((t0+1-1-i):(N-1-i))];  % regressor for input: [u(t0-nk-i), u(t0-nk-i-1), ..., u(N-nk-i)]^T
        end
        Phi = [Yphi, Uphi]; 
        
        Theta = linsolve(Phi'*Phi, Phi'*Y);  % LSE 
        ThetaL = length(Theta); 
        % denDT_hat = [1, Theta(1:na)']; 
        % numDT_hat = Theta(na+1:na+nb)'; 
        % PhatDT = tf(numDT_hat, denDT_hat,  Ts, 'inputdelay', 1);
        Yhat = Phi*Theta; % one-step ahead output prediction
        ehat = Y-Yhat;
        Yhat_loss = sum(ehat.^2)/length(ehat);  % loss function
        Yhat_fit = max(0, 100*(1-norm(ehat)/norm(Y- mean(Y))));  % fitness
        yhat_fit = [yhat_fit;Yhat_fit];
    end
end
% 
% [PhatDT_step, PhatDT_step_t] = step(PhatDT);
% L = length(PhatDT_step_t);
% Yhat_0 = filter([zeros(1, nk+1), numDT_hat], denDT_hat, [uID(1)*ones(2*L,1); uID]); % the first half is used for warming up.
% Yhat_sim0 = Yhat_0; % simulated output
% Yhat_sim = Yhat_sim0(2*L+1:2*L+N);  % select the second half for computation of loss function
% ehat_sim = yID - Yhat_sim;
% Yhat_sim_loss = sum(ehat_sim .^2)/length(ehat);  % loss function
% Yhat_sim_fit = max(0, 100*(1-norm(yID-Yhat_sim)/norm(yID - mean(yID)))); %fitness
    
