Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector
f = 3;                % Signal frequency

S = sin(2*pi*f*t);

Y = fft(S);