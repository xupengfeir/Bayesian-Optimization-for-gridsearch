function [yID, uID] = simModel()

    Ts = 1; % sampling period
    
    %% True System
    num = 1.25; 
    den = [40 1];
    theta = 2; % delay ;
    
    %% PID parameters 
    Kc = 0.7; 
    Ti = 20; 
    Td = 0; 
    alpha = 10; %Anti-derivative kick
       
    %% Process noise configuration
    nvar =1*0.001; 
    n_num = 1;
    n_den = 1; 
    % n_den = [1 -0.9];  % low-pass spectrum noise 
    % n_den = [1 0.5];  % high-pass spectrum noise
    
    %% Define the excitation signal
    r0 = 1*10+ 1*[zeros(1, 900), zeros(1, 130), zeros(1,30), ones(1, 1000), ones(1,40), ones(1, 40), ones(1,40), ones(1, 9000)]';  
    r0_L = length(r0); r0t = linspace(0, Ts*(r0_L-1), r0_L)'; r = [r0t, r0]; 
    
    assignin('base','Ts',Ts);
    assignin('base','num',num);
    assignin('base','den',den);
    assignin('base','theta',theta);
    assignin('base','Kc',Kc);
    assignin('base','Ti',Ti);
    assignin('base','Td',Td);
    assignin('base','alpha',alpha);
    assignin('base','nvar',nvar);
    assignin('base','n_den',n_den);
    assignin('base','n_num',n_num);
    assignin('base','r0_L',r0_L);
    assignin('base','r',r);

    %% ClosedLoopSimulations
    sim('ClosedLoopStepResp_FBD');
    
    %% Prepare the data generated from the simulink model
    u = u0.signals.values;  % controller output
    y = y0.signals.values;  % process output
    r = ref0.signals.values;  % reference
    t = y0.time;
     
%     figure(1), subplot(2,1,1), plot(t, y, 'linewidth', 1), hold on; plot(t, r,'k:', 'linewidth', 2 ); xlabel('time (sec)'), 
%     title('y & r');
%     figure(1), subplot(2,1,2), plot(t, u, 'o', 'linewidth', 1), xlabel('time (sec)'), title('u');
    
    %% Select the data for model ID and validation
    datae_ts = 1000; 
    datae_te = 1400; 
    
    uSel = u([datae_ts:datae_te]); 
    ySel = y([datae_ts:datae_te]); 
    rSel = r0([datae_ts:datae_te]); 
    tSel = t([datae_ts:datae_te]); 
    
%     figure(2), subplot(2,1,1), plot(tSel, ySel, 'linewidth', 1), 
%     hold on; plot(tSel, rSel,'k:', 'linewidth', 2 ); xlabel('time (sec)'), title('y & r'); 
%     subplot(2,1,2), plot(tSel, uSel, 'linewidth', 1), xlabel('time (sec)'), title('u');
    
    %% Estimate the ARX model 
    yID = ySel-10; 
    uID = uSel-4;
