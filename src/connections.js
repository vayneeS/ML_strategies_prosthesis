function isOpen(websocket) {
    return websocket.readyState === websocket.OPEN;
  }

export function sendObj(websocket, packet){
	
	if (!isOpen(websocket)) return;
	websocket.send(JSON.stringify(packet));
}

export async function sendDataToServer(websocket,msg,dataset,phase,trial,predict){
	const instancesdb = {};
	instancesdb['y']=[];
	instancesdb['x']=[];
	const instances = await dataset
	.items()
	.select(['x', 'y']) // select the fields to return
	.toArray();
	for await (const { x, y } of instances) {
		instancesdb['y'].push(y);
		instancesdb['x'].push(x);
	}
	const retrievedData = {
		X_train: instancesdb['x'],
		y_train: instancesdb['y'],
		action: msg,
		phase: phase,
		trial: trial
	};
	console.log("send trainingset to server");
	sendObj(websocket,retrievedData);
	if(predict === true)
		sendObj(websocket,{action: 'prediction',content:{}});
	//sendDbInstances(websocket, instancesdb);
}
// export function sendMsg(websocket, msg) {
// 	const obj = {
// 	  action: msg
// 	};
// 	console.log(msg);
// 	if (!isOpen(websocket)) return;
// 	websocket.send(JSON.stringify(obj));
// } 

export async function retrieveInstancesFromDataset(websocket,msg,trial,dataset) {
	const instances = await dataset
		.items() 
		.select(['x', 'y']) // select the fields to return
		.toArray();
	
	var lbls = []
	var data = []
	for await (const { x, y } of instances) {
		
		// if(itemCounter(lbls, y) < n){
		lbls.push(y)
		data.push(x)
		// }
	}
	console.log(lbls)
	const retrievedData = {
		X_train: data,
		y_train: lbls,
		action: msg,
		trial: trial
	};
	console.log("retrieve data to compute separability");
	sendObj(websocket,retrievedData);
} 	

// export function sendMsg(websocket, msg) {
// 	const obj = {
// 	  action: msg
// 	};
// 	console.log(msg);
// 	if (!isOpen(websocket)) return;
// 	websocket.send(JSON.stringify(obj));
// } 

// export function sendCommands(websocket, cmd){
// 	const retrievedData = {
// 		obj: cmd,
// 		action: 'cmd',
// 	};
// 	console.log(retrievedData);
// 	if (!isOpen(websocket)) return;
// 	websocket.send(JSON.stringify(retrievedData));
// }
// export function sendInstances(websocket, instances){
// 	const retrievedData = {
// 		obj: instances,
// 		action: 'data',
// 	};
// 	console.log(retrievedData);
// 	if (!isOpen(websocket)) return;
// 	websocket.send(JSON.stringify(retrievedData));
// }

// export function sendDbInstances(websocket, instances){
// 	const retrievedData = {
// 		X_train: instances['x'],
// 		y_train: instances['y'],
// 		action: 'data'
// 	};
// 	console.log(retrievedData);
// 	if (!isOpen(websocket)) return;
// 	websocket.send(JSON.stringify(retrievedData));
// }

// export function sendConfidences(websocket, instances){
// 	const retrievedData = {
// 		labels: instances['labels'],
// 		confs: instances['confs'],
// 		action: 'data'
// 	};
// 	console.log(retrievedData);
// 	if (!isOpen(websocket)) return;
// 	websocket.send(JSON.stringify(retrievedData));
// }
const itemCounter = (value, index) => {
    return value.filter((x) => x == index).length;
};

// export async function retrieveInstancesFromDataset(websocket, msg,dataset) {
// 	const n = 1000;
// 	const lastNInstances = {};
// 	const instances = await dataset
// 		.items() 
// 		.query({
// 		$sort: {
// 			createdAt: -1,
// 		},
// 		})
// 		.select(['x', 'y', 'createdAt']) // select the fields to return
// 		.toArray();
// 	//select last n samples of each class
// 	var lbls = []
// 	var data = []
// 	for await (const { x, y, createdAt } of instances) {
		
// 		if(itemCounter(lbls, y) < n){
// 			lbls.push(y)
// 			data.push(x)
// 		}
// 	}
// 	console.log(lbls)
// 	const retrievedData = {
// 		X_train: data,
// 		y_train: lbls,
// 		action: msg,
// 	};
// 	console.log("send trainingset to server");
// 	sendObj(websocket,retrievedData);
// } 	



// export async function getConfidences(websocket,classifier,dataset){
// 	const instancesdb = {};
// 	instancesdb['labels']=[];
// 	instancesdb['confs']=[];
// 	if (!classifier.ready) {
// 		throwError(new Error('No classifier has been trained'));
// 	}
	  
// 	const instances = await dataset
// 	.items()
// 	.select(['x','y']) // select the fields to return
// 	.toArray();
// 	for await (const { x,y } of instances) {
// 		instancesdb['labels'].push(y);
// 		const preds = await classifier.predict(x);
// 		console.log(preds.confidences)
// 		instancesdb['confs'].push(preds.confidences)
	
// 	}
// 	const retrievedData = {
// 		labels: instances['labels'],
// 		confs: instances['confs'],
// 		action: 'data'
// 	};
// 	sendObj(websocket,retrievedData)
// 	//sendConfidences(websocket, instancesdb);
// }