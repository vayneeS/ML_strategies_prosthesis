import '@marcellejs/core/dist/marcelle.css';
import * as marcelle from '@marcellejs/core';
import { Stream } from '@marcellejs/core';

import {sendDataToServer, sendObj,retrieveInstancesFromDataset} from './connections';
import { shuffleArray,getRandomGesture} from './utils';
// import { Predictions } from './components/predictions/predictions.component';
import {thumbnails,gestureSet,gestureSetPosNeg,maxNumInitInstances,maxNumPostTestInstances,maxNumPosNeg,maxNumUserTrainingInstances} from './utils';
import {gestureInit,gesturePosNeg,gestureUserTrain,gesturePostTest} from './utils';
import {store,$features,$pred,$gestureSep,ws} from './utils';
import {dispFamiliarize,getGestureFamiliarize,$gifUserFamiliarize,gestureFilePosNeg,gestureFileInit,gestureFileUserTrain,gestureFilePostTest,gestureFileSep,capturePosNeg,chooseGest,getGestureUserTrain,captureUC,dispUserTrain,predTrain,$imgUserTrain,captureInitAuto,captureNUCSTrain,capturePosNegAuto,capturePosttestAuto,captureNUCTrain} from './utils';

import { dataset } from './components/dataset';

const params = new URLSearchParams(window.location.search);
const pID = params.get('id');
const training = params.get('training');
// let choice = new Stream('',false);

let trainingSet = marcelle.dataset('training-'+pID, store);
let positivesAndNegativesSet = marcelle.dataset('pos-neg-' + pID, store);
let posttestSet = marcelle.dataset('posttest-' + pID, store);

const dataCountInit= dataset(trainingSet);
const dataCountPosNeg= dataset(positivesAndNegativesSet);
const dataCountPostTest= dataset(posttestSet);

// WEBSOCKETS LISTENERS
ws.addEventListener('error', function (event) {
	console.log('Error WebSocket : ', event);
});

ws.addEventListener('message', ({ data }) => {
	const event = JSON.parse(data);
	if (event.features?.length > 0) {
		$features.set(event.features[0]);
	}
	if (event.prediction) {
		$pred.set(event.prediction);
		console.log("server pred: ",event.prediction);
	}
	if (event.sep_gesture) {
		$gestureSep.set(event.sep_gesture);
	}
});

//-------familiarize-----------------//

let counter = 0;

const $pressedGetGestureFamiliarize = getGestureFamiliarize.pressed;
$pressedGetGestureFamiliarize
	.filter((x) => x !== false & counter < gestureSet.length)
	.subscribe(() => {
	const gesture = gestureSet[counter];
	gestureUserTrain.$value.set(gesture);
	$gifUserFamiliarize.set(gesture);
	counter = counter + 1;
	});
	
$gifUserFamiliarize
	.filter((x) => x !== '')
	.map(async (x) => ({
		x,
	}))
	.awaitPromises()
	.subscribe(async (x) => {
		dispFamiliarize.setGIF(x.x);
		dispFamiliarize.setIMG(x.x);
	});
//write random gestures file---------//
let gestureStore = marcelle.dataset('gestures-'+pID,store);

gestureFilePosNeg.$click
	.map(async() => (
		{
			gestures:gestureSetPosNeg,
			user:pID,
			phase : 4
		}))	
	.awaitPromises()
	.subscribe(gestureStore.create);

let g1;
gestureFileInit.$click
	.tap((x) => console.log("x",g1))
	.map(async() => (
		g1 = writeRandomGestures(maxNumInitInstances,gestureSet),
		{
			gestures:g1,
			user:pID,
			phase : 1
		}))	
	.awaitPromises()
	.subscribe(gestureStore.create);


let g2;
gestureFileUserTrain.$click
	.tap((x) => console.log("x",g2))
	.map(async() => (
		g2 = writeRandomGestures(maxNumUserTrainingInstances,gestureSet),
		{
			gestures:g2,
			user:pID,
			phase : 2
		}))	
	.awaitPromises()
	.subscribe(gestureStore.create);


let g3;
gestureFilePostTest.$click
	.tap((x) => console.log("x",g3))
	.map(async() => (
		g3 = writeRandomGestures(maxNumPostTestInstances,gestureSet),
		{
			gestures:g3,
			user:pID,
			phase : 3
		}))	
	.awaitPromises()
	.subscribe(gestureStore.create);

let gsep;
gestureFileSep.$click
	.tap((x) => console.log("x",gsep))
	.map(async() => (
		gsep = writeRandomGestures(maxNumInitInstances,gestureSet),
		{
			gestures:gsep,
			user:pID,
			phase : 2.5
		}))	
	.awaitPromises()
	.subscribe(gestureStore.create);

function writeRandomGestures(maxNumInstances,theGestureSet){
	let indexGesture = Array.from(Array(Math.floor(maxNumInstances)).keys()); 
	let gestures = [];
	shuffleArray(indexGesture);
	for(let i=0;i<maxNumInstances;i++){
		gestures.push(getRandomGesture(indexGesture[i],theGestureSet));
	}
	return gestures;
}

let initGestures = [];
let posNegGestures = [];
let posttestGestures = [];
let userTrainGestures = [];
let sepTrainGestures = [];

gestureStore.$count
	.filter((x) => x > 0)
	.tap((x) => console.log('num files generated: ',x))
	.subscribe(async() => {
		const instances = await gestureStore
		.items()
		.select(['gestures','phase']) // select the fields to return
		.toArray();
		for await (const {gestures,phase} of instances) {
			if(phase == 1)
				initGestures = gestures;
			else if(phase == 2)
				userTrainGestures = gestures;
			else if(phase == 3)
				posttestGestures = gestures;
			else if(phase == 4)
				posNegGestures = gestures;
			else if(phase == 2.5)
				sepTrainGestures = gestures;
			console.log('gestures: ', gestures)
		}
	})
//--------------init trainingset--------------------------//

const $pressedCaptureInitAuto = captureInitAuto.pressed;

const $pressedInitStart = captureInitAuto.start;

$pressedInitStart
	.filter((x) => x !== false)
	.subscribe(async () => {
		console.log('pressed start init')
		setGesture(initGestures,counterGestureInit,captureInitAuto,1)
})

$pressedCaptureInitAuto
  .filter((x) => x !== false)
  .subscribe(async () => {
	ws.send(JSON.stringify('start'));
	console.log('pressed capture init');
  
});

$features
.filter((x) => x.length > 0 && gestureInit.$value.value !== '')
	.subscribe(() => {
		//get next gesture
		setTimeout(function () {
			setGesture(initGestures,counterGestureInit,captureInitAuto,1);
		}, 1000)
})

function setGesture(gestureArray, index, component, phase) {
  const gesture = gestureArray[index];
  console.log('Current gesture:', gesture);

  // Reset all gesture states
  [gestureInit, gesturePosNeg, gestureUserTrain, gesturePostTest].forEach(g => g.$value.set(''));

  if (!gesture) return;

  component.setIMG(gesture);
  switch (phase) {
    case 1: gestureInit.$value.set(gesture); break;
    case 2: gestureUserTrain.$value.set(gesture); break;
    case 3: gesturePostTest.$value.set(gesture); break;
    case 4: gesturePosNeg.$value.set(gesture); break;
  }
}

let counterGestureInit = 0;
trainingSet.$count
	.subscribe((x) => {
		counterGestureInit = Math.floor(x);
		console.log('num trainingset: ',x);
	});


$features
	//check that $features is not empty to prevent error training set not initialized
	.filter((x) => x.length > 0 && gestureInit.$value.value !== '')
	.map(async (x) => ({
		x,
		y: gestureInit.$value.value,
		thumbnail: thumbnails[gestureInit.$value.value],
		condition:training,
	}))
	.awaitPromises()
	.subscribe(trainingSet.create);


trainingSet.$count
	.filter((x) => x >= maxNumInitInstances & gestureInit.$value.value !== '')
	.subscribe(async() => {
		gestureInit.$value.set('');
		captureInitAuto.$completed.set('Completed!');
		sendDataToServer(ws,'train',trainingSet,'phase1',1,false);
		//train classifier before moving to pretest
		console.log("trained");
	});

//---------------phase 4  positives and negatives ------------//


const $pressedCapturePosNeg = capturePosNeg.pressed;
$pressedCapturePosNeg
  .filter((x) => x !== false)
  .subscribe(async () => {
	ws.send(JSON.stringify('start'));
	console.log('pressed capture pretest');
  });

// let indexGesturePreTest = Array.from(Array(Math.floor(maxNumPreTestInstances)).keys()); 
// shuffleArray(indexGesturePreTest);
let counterGesturePosNeg = 0;
positivesAndNegativesSet.$count
	.subscribe((x) => {
		counterGesturePosNeg = Math.floor(x);
	});
const $pressedCapturePosNegAuto = capturePosNegAuto.pressed;
const $pressedPretestStart = capturePosNegAuto.start;
$pressedPretestStart
	.filter((x) => x != false)
	.subscribe(() => {
		setGesture(posNegGestures,counterGesturePosNeg,capturePosNegAuto,4)
	})
$pressedCapturePosNegAuto
	.filter((x) => x != false)
	.subscribe(async() => {
		ws.send(JSON.stringify('start'));
		console.log('pressed capture -ves +ves');
	
	})

$features
	.filter((x) => x.length > 0 && gesturePosNeg.$value.value !== '')
	.subscribe(() => {
		setTimeout(function () {
			setGesture(posNegGestures,counterGesturePosNeg,capturePosNegAuto,4);
		}, 1000)
})


$features
	.filter((x) => x.length > 0 && gesturePosNeg.$value.value !== '')
	.map(async (x) => ({
		x,
		y: gesturePosNeg.$value.value,
		thumbnail: thumbnails[gesturePosNeg.$value.value],
	}))
	.awaitPromises()
	.subscribe(positivesAndNegativesSet.create);

positivesAndNegativesSet.$count
	.filter((x) => x >= maxNumPosNeg)
	.subscribe(() => {
		gesturePosNeg.$value.set('');
		capturePosNegAuto.$completed.set('Completed!');
	});

//--------------------------------user training---------------------
//////User choice////////////
const totalNumTrain = maxNumInitInstances + maxNumUserTrainingInstances;

const $pressedCaptureUserChoice = captureUC.pressed;
const $gestureChoice = captureUC.chosen;
//get gesture choice and set gesture text
$gestureChoice
	.tap((x) => console.log(x))
	.filter((x) => x != '')
	.subscribe(async (x) => {
		gestureUserTrain.$value.set(x);
		gestureInit.$value.set('');
		gesturePosNeg.$value.set('');
		gesturePostTest.$value.set('');
		// choice.set('UC');
	
	});

$pressedCaptureUserChoice
	.filter((x) => x !== false)
	.subscribe(async () => {
		ws.send(JSON.stringify('start'));
	});

const $pressedCaptureTrain = captureNUCTrain.pressed;

$pressedCaptureTrain
	.filter((x) => x !== false)
	.subscribe(async () => {
		ws.send(JSON.stringify('start'));
	})
const $pressedGetGesture = captureNUCTrain.getGesture;

$pressedGetGesture
	.filter((x) => x !== false && training === 'NUC')
	.subscribe(() => {
		const gesture = userTrainGestures[counterGestureInit-maxNumInitInstances];
		console.log('NUC get gesture: ',gesture);
		captureNUCTrain.setIMG(gesture);
		gestureInit.$value.set('');
		gesturePosNeg.$value.set('');
		gestureUserTrain.$value.set(gesture);
		gesturePostTest.$value.set('');
	});


$gestureSep
	.filter((x) => x !== '')
	.subscribe(async (x) => {
		captureNUCSTrain.setIMG(x);
		// dispUserTrain.setIMG(x);
		gestureInit.$value.set('');
		gesturePosNeg.$value.set('');
		gesturePostTest.$value.set('');
		console.log(x)
		gestureUserTrain.$value.set(x);
	})	

const $pressedCaptureNUCS = captureNUCSTrain.pressed;

$pressedCaptureNUCS
	.filter((x) => x !== false)
	.subscribe(async () => {
		ws.send(JSON.stringify('start'));
	})

const $pressedGetGestureNUCS = captureNUCSTrain.getGesture;

$pressedGetGestureNUCS
	.filter((x) => x !== false && training === 'NUCS')
	.subscribe(async() => {
		console.log('next gesture sep');
		retrieveInstancesFromDataset(ws,'sep',counterGestureInit+1,trainingSet);

	});


trainingSet.$count
	.filter((x) => x >= totalNumTrain)
	.subscribe(() => {
		gestureUserTrain.$value.set('');
		dispUserTrain.$completed.set('Completed!');
		captureNUCTrain.$completed.set('Completed!');
		captureNUCSTrain.$completed.set('Completed!');
		captureUC.$completed.set('Completed!');
	});

let $instanceCreated = new Stream(0,false);
trainingSet.$count
	.filter((x)=> x > maxNumInitInstances)
	.subscribe(() =>{
		$instanceCreated.set(counterGestureInit);
	});
//$instanceCreated.subscribe(console.log);
$features
	//check that $features is not empty to prevent error training set not initialized
	.filter((x) => x.length > 0 && gestureUserTrain.$value.value !== '')
	.tap((x) => console.log('getting training features:', x))
	.map(async (x) => ({
		x,
		y: gestureUserTrain.$value.value,
		thumbnail: thumbnails[gestureUserTrain.$value.value],
		condition:training,
	}))
	.awaitPromises()
	.subscribe(trainingSet.create);

console.log('total: ',totalNumTrain);
// Predictions
$instanceCreated
	.filter((x) => x>0)
	//.subscribe(async (feats) => {
	.subscribe(async(x) => {
		//retrain
		sendDataToServer(ws,'train',trainingSet,'phase2',x,true);
		console.log('retrained gesture:',x);
	
	});

$pred
	//.tap((x) => console.log(x))
	.filter((x) => x != undefined)//&& gestureUserTrain.$value.value !== ''
	.subscribe(() => {
		let cmd = getCmd($pred.value,gestureSet);
		if(cmd != undefined){
			const message = {
				action: 'cmd',
				content: cmd
			}
			sendObj(ws,message);
		}
		predTrain.setPredImg($pred.value);
	})
	
function getCmd(predictedLabel,gestures){
	// const expr = 'Rest';
	let cmd = 0;
	switch (predictedLabel) {
		case gestures[1]://'Palm Up'
			cmd = 30;
			break;
		case gestures[0]://'Palm Down'
			cmd = 31;
			break;
		case gestures[3]://'Hand Opening'
			cmd = 1;
			break;
		case gestures[2]://'Hand Closing'
			cmd = 0;
			break;
		case gestures[5]://'Pinch Opening'
			cmd = 11;
			break;
		case gestures[4]://'Pinch Closing'
			cmd = 10;
			break;
		case gestures[6]://'Rest Hand'
			cmd = 50;
			break;
		case gestures[7]://'Index Point'
			cmd = 21;
			break;
		default:
			console.log(`Error got ${predictedLabel}.`);
	}
	console.log('hand cmd:',cmd);
	return cmd;
}

//----------post-test--------------------------------

let counterGesturePostTest = 0;
posttestSet.$count
	.subscribe((x) => {
		counterGesturePostTest = Math.floor(x);
	});


const $pressedPosttestStart = capturePosttestAuto.start;
$pressedPosttestStart
	.filter((x) => x !== false)
	.subscribe(() => {
		console.log('pressed start test')
		setGesture(posttestGestures,counterGesturePostTest,capturePosttestAuto,3)
	})
const $pressedCapturePosttestAuto = capturePosttestAuto.pressed;

$pressedCapturePosttestAuto
	.filter((x) => x !== false)
	.subscribe(() => {
		ws.send(JSON.stringify('start'));
		console.log('pressed capture test');
	
	})
$features
	.filter((x) => x.length > 0 && gesturePostTest.$value.value !== '')
	.subscribe(() => {
		//get next gesture
		setTimeout(function () {
			setGesture(posttestGestures,counterGesturePostTest,capturePosttestAuto,3)
			//console.log("Delayed for x second.");
		}, 1000)
	})


$features
	.filter((x) => x.length > 0 && gesturePostTest.$value.value !== '')
	.map(async (x) => ({
		x,
		y: gesturePostTest.$value.value,
		thumbnail: thumbnails[gesturePostTest.$value.value]
	}))
	.awaitPromises()
	.subscribe(posttestSet.create);


posttestSet.$count
	.filter((x) => x >= maxNumPostTestInstances)
	.subscribe(() => {
		gesturePostTest.$value.set('');
		capturePosttestAuto.$completed.set('Completed!')
	});

//-------------dashboard------------------------------
const dashboard = marcelle.dashboard({
	title: 'Gesture recognition teaching',
	author: '',
  });

dashboard
	.page('Familiarization')
	.use([dispFamiliarize])
	.use([getGestureFamiliarize])

dashboard
	.page('Phase 1')
	.use(captureInitAuto)
	.use(dataCountInit)
	.use(gestureFileInit)
	
console.log(training)

if(training === 'UC'){
  dashboard
  .page('Phase 2')
  .use([captureUC])
  .use(dataCountInit)
  .use(predTrain) 
}
// else if(training === 'M'){
// 	dashboard
// 	.page('Phase 3')
// 	.use([dispUserTrain])
// 	.use([getGestureUserTrain,chooseGest,captureMixed])
// 	.use(dataCountInit)
// 	.use(predTrain)	
// }
else if(training === 'NUC'){
	dashboard
	.page('Phase 2')
	.use(captureNUCTrain)
	.use(dataCountInit)
	.use(gestureFileUserTrain)
	.use(predTrain)
}
else if(training === 'NUCS'){
	dashboard
	.page('Phase 2')
	.use([captureNUCSTrain])
	.use(dataCountInit)
	// .use(gestureFileSep)
	.use(predTrain)
}

dashboard
.page('Phase 3')
// .use([dispPostTest])
.use([capturePosttestAuto])
// .use([getGesturePostTest,captureDataPostTest])
.use(dataCountPostTest)
.use(gestureFilePostTest)

dashboard
	.page('Phase 4')
	.use([capturePosNegAuto])
	.use(dataCountPosNeg)
	.use(gestureFilePosNeg)

dashboard.settings.dataStores(store).datasets(trainingSet,posttestSet,positivesAndNegativesSet);
// dashboard.settings.dataStores(store).datasets(posttestSet);
// dashboard.settings.dataStores(store).datasets(positivesAndNegativesSet);

dashboard.show();
