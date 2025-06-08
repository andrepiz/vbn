%% VBN INSTALLATION %%

% MATLAB environment
clear
clc
close all

if isfile("vbn_install.m")
    % SAVE INSTALL TO PATH
    copyfile("vbn_install.m",userpath)
    % SAVE PATH TREE
    writelines(["function vbn_home_path = vbn_home()",join(["vbn_home_path = '",pwd,"';"],''),"end"],'vbn_home.m')
    movefile('vbn_home.m',userpath)
end

% Path
addpath(genpath(vbn_home))
rmpath(genpath(fullfile(vbn_home,'debug')))

% DISPLAY
fprintf(['*** VBN installed. Have fun! ***\n'])