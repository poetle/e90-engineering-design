%% PD and ET tremor analysis
%% by Jackie Le


%% note: code below computations section shows a longstanding hsitory
%% of the various approaches taken to qualitatively and quantiatively
%% explore tremors of various types. as such, numerous algorithms and
%% plots are attempted to perform this objective.

%% reset
close all; clear; clc;

%% load file accordingly
file_name = 'real_tremor_datasets/Park_L_RT_1.txt';
% file_name = 'imu_data.txt';
file_id = fopen(file_name);

% read file
dot_h = 10; % header lines
% dot_h = 1; % header lines
h_data = textscan(file_id, '%s', dot_h, 'Delimiter', '\n');
data = textscan(file_id, '%f %f %f %f', 'Delimiter', ' ');

% parse tremor data
og_a_x = data{1}; % x acceleration
og_a_y = data{2}; % y acceleration
og_a_z = data{3}; % z acceleration



%% recalibrate to raw acceleration
g = 9.81; % m/s^2
mVperg = 6; Vperg = mVperg/1000; % mV to V
raw_a_x = og_a_x*g/Vperg; raw_a_y = og_a_y*g/Vperg; raw_a_z = og_a_z*g/Vperg;

% define time vector (constant sampling frequency)
n_rows = size(data{1},1);
f_s = 200; % sampling frequency (Hz)
t = (0:n_rows-1)*(1/f_s); % time (s)



%% newtonian mechanics
% integrate -> velocity
dt = mean(diff(t)); % uniform sampling
v_x = cumsum(raw_a_x)*dt; v_y = cumsum(raw_a_y)*dt; v_z = cumsum(raw_a_z)*dt;

% integrate -> displacement
dst_x = cumsum(v_x)*dt; dst_y = cumsum(v_y)*dt; dst_z = cumsum(v_z)*dt;

% m to cm for tremor scale
x_dst_cm = dst_x*100; y_dst_cm = dst_y*100; z_dst_cm = dst_z*100;



%% frequency plots
% fast fourier transform raw acceleration, velocity, displacement
fft_raw_a_x = fft(raw_a_x); fft_raw_a_y = fft(raw_a_y); fft_raw_a_z = fft(raw_a_z);
fft_x_dst = fft(dst_x); fft_y_dst = fft(dst_y); fft_z_dst = fft(dst_z);
fft_v_x = fft(v_x); fft_v_y = fft(v_y); fft_v_z = fft(v_z);

% single_sided frequency vector
f_sngsd = (0:n_rows/2)*f_s/n_rows;

% define the frequency range of interest (Hz)
pd_f_r = [4,6]; % PD peaks ~5 Hz
et_f_r = [4,12]; % ET peaks ~8-10 Hz

% file-conditional tremor frequency range
if contains(file_name,"PD")
    f_r = pd_f_r;
elseif contains(file_name,"ET")
    f_r = et_f_r;
else
   f_r = [0,20];
end



%% peak detection
% corresponding frequency vector indices
f_idx_i = find(f_sngsd >= f_r(1),1);
f_idx_f = find(f_sngsd <= f_r(2),1,'last');

% thresholding within frequency range
thrsh = 0.5; % Adjust this thrsh value as needed
idx_x_pks = find(abs(fft_raw_a_x(f_idx_i:f_idx_f)) > thrsh);
idx_y_pks = find(abs(fft_raw_a_y(f_idx_i:f_idx_f)) > thrsh);
idx_z_pks = find(abs(fft_raw_a_z(f_idx_i:f_idx_f)) > thrsh);

% adjust peak indices to global frequency indices (f_r)
idx_x_pks = idx_x_pks + f_idx_i - 1;
idx_y_pks = idx_y_pks + f_idx_i - 1;
idx_z_pks = idx_z_pks + f_idx_i - 1;

% trace peak values from peak indices
val_x_pk = abs(fft_raw_a_x(idx_x_pks));
val_y_pk = abs(fft_raw_a_y(idx_y_pks));
val_z_pk = abs(fft_raw_a_z(idx_z_pks));

% find max peak frequencies [plot(x_pk_f,max_x,"ro");]
[max_x,idx_x_pk] = max(val_x_pk); x_pk_f = f_sngsd(idx_x_pks(idx_x_pk));
[y_max,idx_y_pk] = max(val_y_pk); y_pk_f = f_sngsd(idx_y_pks(idx_y_pk));
[z_max,idx_z_pk] = max(val_z_pk); z_pk_f = f_sngsd(idx_z_pks(idx_z_pk));



%% filtering: remove pure DC, maximize tremor noise, keep to scale
f_c_lst = [0.5,0.1,0.3]; ord_filt = 1;
figure(10);
for i = 1:size(f_c_lst,2)
    f_c = f_c_lst(i); % higher cutoff frequency-> lower amplitude
    [b,a] = butter(ord_filt,f_c/(f_s/2),'high'); % high-pass

    % filter x,y,z displacement
    filt_x_dsp = filtfilt(b,a,x_dst_cm);
    filt_y_dsp = filtfilt(b,a,y_dst_cm);
    filt_z_dsp = filtfilt(b,a,z_dst_cm);

    plot(t,filt_x_dsp)
    hold on
end
xlabel("Time (seconds)"); ylabel("Filtered x displacement (centimeters)");
title("Filtered displacement versus time");
lgnd_lbls = arrayfun(@num2str,f_c_lst,'UniformOutput',false); legend(lgnd_lbls);

% fft_filt_x_dsp = fft(filt_x_dsp); fft_filt_y_dsp = fft(filt_y_dsp); fft_filt_z_dsp = fft(filt_z_dsp);

% design high-pass filter
Fs = 200;  % sampling frequency

Fstop = 0.1;               % stopband frequency
Fpass = 0.3;               % passband frequency
Dstop = 0.001;             % stopband attenuation
Dpass = 0.017267671642;    % passband ripple
dens  = 20;                % density factor

% calculate order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop, Fpass]/(Fs/2), [0 1], [Dstop, Dpass]);

% calculate coefficients using FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
a = 1; % FIr numerator








% plot filtered velocity
figure(100);
subplot(3,1,1);
filt_v_x = filtfilt(b,a,v_x);
plot(t,v_x);
subplot(3,1,2);
filt_v_y = filtfilt(b,a,v_y);
plot(t,v_y);
subplot(3,1,3);
filt_v_z = filtfilt(b,a,v_z);
plot(t,v_z);

filt_dst_x = cumsum(filt_v_x)*dt; filt_dst_y = cumsum(filt_v_y)*dt; filt_dst_z = cumsum(filt_v_z)*dt;

% m to cm for tremor scale
filt_x_dst_cm = filt_dst_x*100; filt_y_dst_cm = filt_dst_y*100; filt_z_dst_cm = filt_dst_z*100;

figure(99);
subplot(3,1,1);
plot(t,filt_x_dst_cm);
title("Filtered displacement versus time");
ylabel("x (centimeters)");
% xlim([25,35])
subplot(3,1,2);
plot(t,filt_y_dst_cm);
ylabel("y (centimeters)")
% xlim([25,35])
subplot(3,1,3);
plot(t,filt_z_dst_cm);
ylabel("z (centimeters)");xlabel("Time (seconds)")
% xlim([25,35])

figure(11);
mag_xyz_dst_cm = sqrt((filt_x_dst_cm-mean(filt_x_dst_cm)).^2+(filt_y_dst_cm-mean(filt_y_dst_cm)).^2+(filt_z_dst_cm-mean(filt_z_dst_cm)).^2);
mag_xyz_dst_cm2 = sqrt(filt_x_dst_cm.^2+filt_y_dst_cm.^2+filt_z_dst_cm.^2);
fft_mag_xyz_dst_cm = fft(mag_xyz_dst_cm);
fft_mag_xyz_dst_cm2 = fft(mag_xyz_dst_cm2);
plot(t,mag_xyz_dst_cm);
% plot(f_sngsd,abs(fft_mag_xyz_dst_cm(1:n_rows/2+1)));
% plot(f_sngsd,abs(fft_mag_xyz_dst_cm2(1:n_rows/2+1)));
title("Magnitude of filtered displacement versus time");
xlabel("Time (seconds)"); ylabel("x, y, z (centimeters)");

% plot velocity versus time
figure(2);
subplot(3,1,1);
plot(t,v_x);
title("Velocity versus time")
ylabel("x' (meters/second)")
subplot(3,1,2);
plot(t,v_y);
ylabel("y' (meters/second)")
subplot(3,1,3);
plot(t,v_z);
xlabel("Time (seconds)")
ylabel("z' (meters/second)")

% plot displacement vs frequency
figure(1000); subplot(3,1,1); plot(f_sngsd,abs(fft_x_dst(1:n_rows/2+1)));
xlim([0 20]);
hold on; ylabel("x magnitude");
title("Raw displacement frequency spectrum")

subplot(3,1,2); plot(f_sngsd, abs(fft_y_dst(1:n_rows/2+1)));
xlim([0 20]);
hold on; ylabel("y magnitude");

subplot(3,1,3); plot(f_sngsd,abs(fft_z_dst(1:n_rows/2+1)));
xlim([0 20]);
hold on; xlabel("Frequency (hertz)"); ylabel("z magnitude");

% plot raw acceleration vs time
figure(1); subplot(3,1,1); plot(t,raw_a_x);
ylabel("x'' (meters/second^2)");
title('Raw acceleration versus time')

subplot(3,1,2); plot(t,raw_a_y);
ylabel("y'' (meters/second^2)");

subplot(3,1,3); plot(t,raw_a_z);
xlabel("Time (seconds)"); ylabel("z'' (meters/second^2)");

% plot raw velocity vs time
figure(90); subplot(3,1,1); plot(t,v_x);
ylabel("x' (meters/second)");
title("Velocity versus time");

subplot(3,1,2); plot(t,v_y);
ylabel("y' (meters/second)");

subplot(3,1,3); plot(t,v_z);
xlabel("Time (seconds)"); ylabel("z' (meters/second)");

% plot filtered displacement vs time
figure(2); subplot(3,1,1); plot(t,filt_x_dsp);
ylabel("Filtered x (centmeters)");
title("Filtered displacement versus time")

subplot(3,1,2); plot(t,filt_y_dsp);
ylabel("Filtered y (centmeters)");

subplot(3,1,3); plot(t,filt_z_dsp);
xlabel("Time (seconds)"); ylabel("Filtered z (centmeters)");

% plot displacement vs time
figure(3); subplot(3,1,1); plot(t,x_dst_cm);
ylabel("x (centimeters)");
title("Displacement versus time")

subplot(3,1,2); plot(t,y_dst_cm);
ylabel("y (centimeters)");

subplot(3,1,3); plot(t,z_dst_cm);
xlabel("Time (seconds)"); ylabel("z (centimeters)");

% plot magnitude displacement vs time
filt_xyz_dst_1 = sqrt(filt_x_dsp.^2+filt_y_dsp.^2+filt_z_dsp.^2);
fft_filt_xyz_dst_1 = fft(filt_xyz_dst_1);
fft_filt_xyz_dst_2 = sqrt(fft_filt_x_dsp.^2+fft_filt_y_dsp.^2+fft_filt_z_dsp.^2);

figure(4); plot(t,filt_xyz_dst_1); hold on;
xyz_dst = sqrt(x_dst_cm.^2+y_dst_cm.^2+z_dst_cm.^2);
filt_xyz_dst_2 = filtfilt(b,a,xyz_dst);
plot(t,filt_xyz_dst_2);
xlabel("Time (seconds)"); ylabel("Filtered magnitude displacement x,y,z (cm)");
title("Magnitide versus time")

xyz_dst_1 = sqrt(x_dst_cm.^2+y_dst_cm.^2+z_dst_cm.^2);
fft_xyz_dst_1 = fft(xyz_dst_1);

figure(11); plot(t,xyz_dst_1);
xlabel("Time (seconds)"); ylabel("Raw x, y, z magnitude displacement (centimeters)")

% plot raw acceleration vs frequency (single-sided)
figure(9); subplot(3,1,1); plot(f_sngsd,abs(fft_raw_a_x(1:n_rows/2+1)));
hold on; ylabel("x'' magnitude");
plot(x_pk_f,max_x,"ro");
xline(4);
xline(12);
xlim([0 20]);
title("Raw acceleration frequency spectrum")

subplot(3,1,2); plot(f_sngsd, abs(fft_raw_a_y(1:n_rows/2+1)));
xlim([0 20]);
xline(4);
xline(12);
hold on; ylabel("y'' magnitude");
plot(y_pk_f,y_max,"ro");

subplot(3,1,3); plot(f_sngsd,abs(fft_raw_a_z(1:n_rows/2+1)));
xlim([0 20]);
xline(4);
xline(12);
hold on; xlabel("Frequency (hertz)"); ylabel("z'' magnitude");
plot(z_pk_f, z_max, "ro");

% plot velocity vs frequency (single-sided)
figure(6); subplot(3,1,1); plot(f_sngsd,abs(fft_v_x(1:n_rows/2+1)));
hold on
ylabel("x' magnitude");
xlim([0 20]);
title("Velocity frequency spectrum")

subplot(3,1,2); plot(f_sngsd,abs(fft_v_y(1:n_rows/2+1)));
xlim([0 20]);
ylabel("y' magnitude");

subplot(3,1,3); plot(f_sngsd,abs(fft_v_z(1:n_rows/2+1)));
xlim([0 20]);
xlabel("Frequency (hertz)"); ylabel("z' magnitude");

% plot filtered displacement vs frequency (single-sided)
% normalize by signal length
figure(7); subplot(3,1,1); plot(f_sngsd,abs(fft_filt_x_dsp(1:n_rows/2+1)));
xlim([0 20]);
ylabel("x magnitude");
title("Displacement frequency spectrum");

subplot(3,1,2); plot(f_sngsd,abs(fft_filt_y_dsp(1:n_rows/2+1)));
xlim([0 20]);
ylabel("y magnitude");

subplot(3,1,3); plot(f_sngsd,abs(fft_filt_z_dsp(1:n_rows/2+1)));
xlim([0 20]);
ylabel("z magnitude");

figure(8); plot(f_sngsd,abs(fft_filt_xyz_dst_1(1:n_rows/2+1)));
hold on; % plot(f_sngsd,fft_filt_xyz_dst_2(1:n_rows/2+1));
xlabel("Frequency (hertz)"); ylabel('xyz displacement magnitude');
xlim([0 12]); ylim([0 100]);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% attempted approaches
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% polynomial fit deduction (runtime costly)
% windowing
gfit_xyz_dst_1 = smoothdata(xyz_dst_1,'gaussian',SmoothingFactor=0.04);
figure(18); plot(t,gfit_xyz_dst_1);
xlabel("Time (seconds)"); ylabel("Fit magnitude displacement x,y,z (centimeters)")
dc_filt_xyz_dst_1 = gfit_xyz_dst_1-xyz_dst_1;
figure(19); plot(t,dc_filt_xyz_dst_1); hold on; plot(t,filt_xyz_dst_1)
xlabel("Time (seconds)"); ylabel("Filtered x, y, z displacement (centimeters)");

f_c2 = 0.4; ord_filt = 1;
[b,a] = butter(ord_filt,f_c/(f_s/2),'high'); % high-pass
hpf_dc_filt_xyz_dst_1 = filtfilt(b,a,dc_filt_xyz_dst_1);
figure(20); plot(t,hpf_dc_filt_xyz_dst_1);
xlabel("Time (seconds)"); ylabel("High-pass filtered x,y,z displacement (centimeters)");

fft_dc_filt_xyz_dst_1 = fft(dc_filt_xyz_dst_1);
figure(21); plot(f_sngsd,abs(fft_dc_filt_xyz_dst_1(1:n_rows/2+1)));
hold on;
xlabel("Frequency (hertz)"); ylabel("High-pass filtered x,y,z magnitude");
xlim([0 12]); ylim([0 100]);



%% wavelet decomposition (data too nonoscillatory, and runtime high)
wave_lvl = 5;
[xyz_wave_coef,~] = wavedec(xyz_dst_1,wave_lvl,'db4');
figure(16);
for i = 1:wave_lvl+1
    subplot(wave_lvl+1,1,i);
    plot(t,xyz_wave_coef(i,:));
    title(["Level ", num2str(i-1)]);
end