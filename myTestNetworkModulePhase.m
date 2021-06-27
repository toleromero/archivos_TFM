function [denoisedAudio] = myTestNetworkModulePhase(noisyAudio)

    Fs=8000;
	
    WindowLength = 256; % 20ms*Fs = 160 --> 2^8 = 256 --> 256 > 160
	win = hamming(WindowLength,"periodic"); % Hamming window
	overlap = round(0.75 * WindowLength); % 75% overlap
	fftLength = WindowLength;
	numFeatures = fftLength/2 + 1;
	numSegments = 8;

	% Training module and phase together
    s = load("myDenoisednetModulePhase.mat"); 
    denoiseNetFullyConvolutional = s.denoiseNetFullyConvolutional_ModulePhase;
    cleanMean = s.cleanMean;
    cleanStd = s.cleanStd;
    noisyMean = s.noisyMean;
    noisyStd = s.noisyStd;
    
    noisySTFT = spectrogram(noisyAudio, win, overlap, fftLength, Fs, 'yaxis');
    noisySTFT_m = abs(noisySTFT)'; % Transpose to have time in horizontal axis
    noisySTFT_p = unwrap(angle(noisySTFT))'; % Transpose to have time in horizontal axis
    noisySTFT_p = noisySTFT_p./300; % Normalize because phase are values too huge compared to the module

    noisySTFT_all = [noisySTFT_m, noisySTFT_p]; % Concat module and phase
    noisySTFT=noisySTFT_all;

    %Generate the 8-segment training predictor signals from the noisy STFT. The overlap between consecutive predictors is 7 segments
    noisySTFT = [noisySTFT(:,1:numSegments-1) noisySTFT];
	predictors = zeros(size(noisySTFT,1), numSegments , size(noisySTFT,2) - numSegments + 1);
    for index = 1:(size(noisySTFT,2) - numSegments + 1)
		predictors(:,:,index) = noisySTFT(:,index:index + numSegments - 1); 
    end

    %Normalize the predictors by the mean and standard deviation computed in the training stage
    predictors(:) = (predictors(:) - noisyMean) / noisyStd;

    %Compute the denoised magnitude STFT by using predict with the two trained networks
    predictors = reshape(predictors, size(predictors,3),size(predictors,2),1,size(predictors,1));
    STFTFullyConvolutional = predict(denoiseNetFullyConvolutional, predictors);

    %Scale the outputs by the mean and standard deviation used in the training stage
    STFTFullyConvolutional(:) = cleanStd * STFTFullyConvolutional(:) + cleanMean;

    %Convert the one-sided STFT to a centered STFT.
    STFTFullyConvolutional = squeeze(STFTFullyConvolutional);
    modulo = STFTFullyConvolutional(1:numFeatures,:); % Part 1 following concat in line 25
    fase = STFTFullyConvolutional(numFeatures+1:end,:)*300; % Part 2 following concat in line 25 and recovering the original value multiplying by 300
    fase_pi=wrapToPi(fase);

    STFTFullyConvolutional = modulo.*exp(1j*fase_pi);
    STFTFullyConvolutional = [conj(STFTFullyConvolutional(end-1:-1:2,:)) ; STFTFullyConvolutional];

    %Compute the denoised speech signals                               
    denoisedAudio = istft(STFTFullyConvolutional,  ...
                                            'Window',win,'OverlapLength',overlap, ...
                                            'FFTLength',fftLength,'ConjugateSymmetric',true);
    sound(denoisedAudio,Fs);
end