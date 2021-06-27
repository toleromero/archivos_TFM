function [denoisedAudio] = myTestNetworkPhase_no_unwrap_threshold(noisyAudio)

    Fs=8000;
	
    WindowLength = 256; % 20ms*Fs = 160 --> 2^8 = 256 --> 256 > 160
	win = hamming(WindowLength,"periodic"); % Hamming window
	overlap = round(0.75 * WindowLength); % 75% overlap
	fftLength = WindowLength;
	numFeatures = fftLength/2 + 1;
	numSegments = 8;
 
	% Use stft to generate magnitude STFT vectors from the noisy noisyAudio signals
	noisySTFT = stft(noisyAudio,'Window',win,'OverlapLength',overlap,'FFTLength',fftLength);
	noisyPhase_original = angle(noisySTFT(numFeatures-1:end,:));
	noisySTFT = abs(noisySTFT(numFeatures-1:end,:));

	%% MODULE
	% Training module
    m = load("myDenoisenet.mat");
    denoiseNetFullyConvolutionalModule = m.myDenoiseNetFullyConvolutional;
		
	cleanMean = m.cleanMean;
    cleanStd = m.cleanStd;
    noisyMean = m.noisyMean;
    noisyStd = m.noisyStd;
	
	% Generate the 8-segment training predictor signals from the noisy STFT. The overlap between consecutive predictors is 7 segments
	noisySTFT = [noisySTFT(:,1:numSegments-1) noisySTFT];
	predictors = zeros( numFeatures, numSegments , size(noisySTFT,2) - numSegments + 1);
	for index = 1:(size(noisySTFT,2) - numSegments + 1)
		predictors(:,:,index) = noisySTFT(:,index:index + numSegments - 1); 
	end
	
	%Normalize the predictors by the mean and standard deviation computed in the training stage
	predictors(:) = (predictors(:) - noisyMean) / noisyStd;

	%Compute the denoised magnitude STFT by using predict with the two trained networks
	predictors = reshape(predictors, [numFeatures,numSegments,1,size(predictors,3)]);
	STFTFullyConvolutional = predict(denoiseNetFullyConvolutionalModule, predictors);
	
	%Scale the outputs by the mean and standard deviation used in the training stage
	STFTFullyConvolutional(:) = cleanStd * STFTFullyConvolutional(:) + cleanMean;
	
	STFTFullyConvolutional = squeeze(STFTFullyConvolutional);
	STFTFullyConvolutional = [conj(STFTFullyConvolutional(end-1:-1:2,:)) ; STFTFullyConvolutional];
	
	
	%% PHASE
	% Training phase (no unwrap)
	p = load("myDenoisenetPhase_no_unwrap.mat");
    denoiseNetFullyConvolutionalPhase = p.denoiseNetFullyConvolutionalPhase_no_unwrap;
        
    cleanMeanPhase = p.cleanMean;
    cleanStdPhase = p.cleanStd;
    noisyMeanPhase = p.noisyMean;
    noisyStdPhase = p.noisyStd;
    
	% Generate the 8-segment training predictor signals from the noisy STFT. The overlap between consecutive predictors is 7 segments
	noisyPhase = [noisyPhase_original(:,1:numSegments-1) noisyPhase_original];
	predictorsPhase = zeros( numFeatures, numSegments , size(noisyPhase,2) - numSegments + 1);
	for index = 1:(size(noisyPhase,2) - numSegments + 1)
		predictorsPhase(:,:,index) = noisyPhase(:,index:index + numSegments - 1); 
	end
	
    predictorsPhase(:) = (predictorsPhase(:) - noisyMeanPhase) / noisyStdPhase;
    
    predictorsPhase = reshape(predictorsPhase, [numFeatures,numSegments,1,size(predictorsPhase,3)]);
	STFTFullyConvolutionalPhase = predict(denoiseNetFullyConvolutionalPhase, predictorsPhase);

    STFTFullyConvolutionalPhase(:) = cleanStdPhase * STFTFullyConvolutionalPhase(:) + cleanMeanPhase;
 
    STFTFullyConvolutionalPhase = squeeze(STFTFullyConvolutionalPhase);
	STFTFullyConvolutionalPhase = [conj(STFTFullyConvolutionalPhase(end-1:-1:2,:)) ; STFTFullyConvolutionalPhase];
    
	%% RESULT
	module = squeeze(STFTFullyConvolutional);
	
	threshold=-15; %dB
	fase_ok=zeros(size(module));
    for i=1: length(module)
        for j=1: 129
            if 20*log10(abs(module(j,i))) > threshold
                fase_ok(j,i) = noisyPhase_original(j,i);
            else
                fase_ok(j,i) = STFTFullyConvolutionalPhase(j,i);   
            end
        end
    end	
	
	STFTFullyConvolutional_complete = STFTFullyConvolutional.*exp(1j*STFTFullyConvolutionalPhase);

	%Compute the denoised speech signals                               
    denoisedAudio = istft(STFTFullyConvolutional_complete,  ...
                                        'Window',win,'OverlapLength',overlap, ...
                                        'FFTLength',fftLength,'ConjugateSymmetric',true);
	
    sound(denoisedAudio,Fs);
    
end