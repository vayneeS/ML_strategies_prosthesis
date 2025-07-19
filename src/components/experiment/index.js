import { Experiment } from './experiment.component';

export function experiment(...args) {
	return new Experiment(...args);
}
