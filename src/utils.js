import '@marcellejs/core/dist/marcelle.css';
import * as marcelle from '@marcellejs/core';
import { Stream } from '@marcellejs/core';
import { userChoiceCaptureEmg,userChoice,noUserChoiceTrainCaptureEmg,gestureDisplay,getGesture, predictions,noUserChoiceCaptureEMG, captureEmg } from './components';

export const ws = new WebSocket('ws://127.0.0.1:8765/');

export function shuffleArray(array) {
	for (let i = array.length - 1; i > 0; i--) {
	  const j = Math.floor(Math.random() * (i + 1));
	  if(array[i]%8 != array[j+1]%8 && array[i]%8 != array[j-1]%8 && array[j]%8 != array[i-1]%8 && array[j]%8 != array[i+1]%8)
	    [array[i], array[j]] = [array[j], array[i]];
	}
	return array; 
}
export async function writeLog(service, data){
  if(data != undefined){
    service.create(data);
  }
}

export function getRandomGesture(index, theGestureSet){

	return theGestureSet[index % 8];
}

export async function retrieveRandomGestures(pID,theService){
	const retrievedGestures = await theService
		.items()
		.query({	
				user: pID	
			})
		.select(['gestures'])
		.toArray()
    .awaitPromises() 
}

export const thumbnails = {
  'Palm Down': '../images/wp.svg',
  'Palm Up': '../images/ws.svg',
  'Close Hand': '../images/hc.svg',
  'Open Hand': '../images/ho.svg',
  'Close Pinch': '../images/pc.svg',
  'Open Pinch': '../images/po.svg',
  'Rest Hand': '../images/rt.svg',
  'Point Index': '../images/ip.svg',
};

export const thumbnail_pos_neg = {
  'Palm Down Negative': '../images/wp.svg',
  'Palm Down Positive': '../images/wp.svg',
  'Palm Up Negative': '../images/ws.svg',
  'Palm Up Positive': '../images/ws.svg',
  'Close Hand Negative': '../images/hc.svg',
  'Close Hand Positive': '../images/hc.svg',
  'Open Hand Negative': '../images/ho.svg',
  'Open Hand Positive': '../images/ho.svg',
  'Close Pinch Negative': '../images/pc.svg',
  'Close Pinch Positive': '../images/pc.svg',
  'Open Pinch Negative': '../images/po.svg',
  'Open Pinch Positive': '../images/po.svg',
  'Rest Hand Negative': '../images/rt.svg',
  'Rest Hand Positive': '../images/rt.svg',
  'Point Index Negative': '../images/ip.svg',
  'Point Index Positive': '../images/ip.svg',
};

export const gifs = {
  'Palm Down': '../images/wp.gif',
  'Palm Up': '../images/ws.gif',
  'Close Hand': '../images/hc.gif',
  'Open Hand': '../images/ho.gif',
  'Close Pinch': '../images/pc.gif',
  'Open Pinch': '../images/po.gif',
  'Rest Hand': '../images/rt.gif',
  'Point Index': '../images/ip.gif',
};

export const gestureSet= [
	'Palm Down', 'Palm Up', 'Close Hand',
	'Open Hand', 'Close Pinch', 'Open Pinch', 'Rest Hand', 'Point Index'
];

export const gestureSetPosNeg= [];
for(var i=0;i<8;i++){
  for(var j=0;j<3;j++){
    gestureSetPosNeg.push(gestureSet[i]+' Positive');
  }
  for(var j=0;j<3;j++){
    gestureSetPosNeg.push(gestureSet[i]+' Negative');
  }
}
export const maxNumPosNeg = 48;///48
export const maxNumInitInstances = 16;//16
export const maxNumUserTrainingInstances = 104;//104 
export const maxNumPostTestInstances = 48;//48

export const gestureInit = marcelle.text('');
export const gesturePosNeg = marcelle.text('');
export const gestureUserTrain = marcelle.text('');
export const gesturePostTest = marcelle.text('');

export const store = marcelle.dataStore('http://localhost:3030');
export const $features = new Stream([], false);
export const $pred = new Stream('',false);
export const $gestureSep = new Stream('',false);

export const dispFamiliarize = gestureDisplay();
export const getGestureFamiliarize = getGesture();
export const $gifUserFamiliarize = new Stream('', false);

export const gestureFilePosNeg = marcelle.button('Generate file');
gestureFilePosNeg.title = '';

export const gestureFileInit = marcelle.button('Generate file');
gestureFileInit.title = '';

export const gestureFileSep = marcelle.button('Generate file');
gestureFileSep.title = '';


export const gestureFileUserTrain = marcelle.button('Generate file');
gestureFileUserTrain.title = '';

export const gestureFilePostTest = marcelle.button('Generate file');
gestureFilePostTest.title = '';

export const start = marcelle.button('Start Experiment');
start.title = '';

export const dispPosNeg = gestureDisplay();
export const capturePosNeg = userChoiceCaptureEmg(); 

export const chooseGest = userChoice();
export const getGestureUserTrain = getGesture();
export const captureUC = userChoiceCaptureEmg();

export const dispUserTrain = gestureDisplay();
export const predTrain = predictions();
export const $imgUserTrain = new Stream('', false);

export const captureInitAuto = noUserChoiceCaptureEMG();
export const capturePosNegAuto = noUserChoiceCaptureEMG();
export const capturePosttestAuto = noUserChoiceCaptureEMG();
export const captureMixed = captureEmg();
export const captureNUCSTrain = noUserChoiceTrainCaptureEmg();
export const captureNUCTrain = noUserChoiceTrainCaptureEmg();