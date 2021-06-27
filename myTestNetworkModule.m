function [denoisedAudio] = myTestNetworkModule(noisyAudio)

    Fs=8000;
	
    WindowLength = 256; % 20ms*Fs = 160 --> 2^8 = 256 --> 256 > 160
	win = hamming(WindowLength,"periodic"); % Hamming window
	overlap = round(0.75 * WindowLength); % 75% overlap
	fftLength = WindowLength;
	numFeatures = fftLength/2 + 1;
	numSegments = 8;

	% Training only the module
    s = load("myDenoisenet.mat");
    denoiseNetFullyConvolutional = s.myDenoiseNetFullyConvolutional;
    cleanMean = s.cleanMean;
    cleanStd = s.cleanStd;
    noisyMean = s.noisyMean;
    noisyStd = s.noisyStd;
    
	%Use stft to generate magnitude STFT vectors from the noisy noisyAudio signals
	noisySTFT = stft(noisyAudio,'Window',win,'OverlapLength',overlap,'FFTLength',fftLength);
	noisyPhase = angle(noisySTFT(numFeatures-1:end,:));
	noisySTFT = abs(noisySTFT(numFeatures-1:end,:));

	% Generate the 8-segment training predictor signals from the noisy STFT. The overlap between consecutive predictors is 7 segments
	noisySTFT = [noisySTFT(:,1:numSegments-1) noisySTFT];
	predictors = zeros( numFeatures, numSegments , size(noisySTFT,2) - numSegments + 1);
	for index = 1:(size(noisySTFT,2) - numSegments + 1)
		predictors(:,:,index) = noisySTFT(:,index:index + numSegments - 1); 
	end

	% Normalize the predictors by the mean and standard deviation computed in the training stage
	predictors(:) = (predictors(:) - noisyMean) / noisyStd;

	% Compute the denoised magnitude STFT by using predict with the two trained networks
	predictors = reshape(predictors, [numFeatures,numSegments,1,size(predictors,3)]);
	STFTFullyConvolutional = predict(denoiseNetFullyConvolutional, predictors);

	% Scale the outputs by the mean and standard deviation used in the training stage
	STFTFullyConvolutional(:) = cleanStd * STFTFullyConvolutional(:) + cleanMean;

	% Convert the one-sided STFT to a centered STFT
	STFTFullyConvolutional = squeeze(STFTFullyConvolutional) .* exp(1j*noisyPhase);
	STFTFullyConvolutional = [conj(STFTFullyConvolutional(end-1:-1:2,:)) ; STFTFullyConvolutional];

	% Compute the denoised speech signals
	denoisedAudio = istft(STFTFullyConvolutional,  ...
											'Window',win,'OverlapLength',overlap, ...
											'FFTLength',fftLength,'ConjugateSymmetric',true);
                                        
	sound(denoisedAudio,Fs);
end