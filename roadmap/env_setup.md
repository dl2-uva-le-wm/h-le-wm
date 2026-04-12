```
### Environment Setup
- [ ] Find source for the Humanoid dataset (state-action-reward dataset)
* [ ] Understand environments that are compatible with the LeWorldModel setup.
	* [ ] PushT
	* [ ] TwoRooms
	* [ ] Man\ipulator
* [ ] Understand if we can use the Humanoid env out-of-the-box or if the construction of a dataset is required.
	* [ ] Construct the Humanoid dataset as the others and add them on Hugging Face consistently with the original paper.
* [ ] Once the environment is understood and working
	* [ ] Provide an API for interacting with it easily
	* [ ] Write a README.md.

### Evaluation & Benchmarking

#### Evaluation Pipeline
* [ ] Integrate the `stable-worldmodel` Python library to establish the base training and evaluation loop.
* [ ] Implement evaluation scripts to measure success rates and prediction accuracy across the three temporal levels ($1$, $k_{1}$, $k_{2}$).
	* [ ] Define what the evaluation scripts will do
	* [ ] Write them and test them
* [ ] Design and run ablation studies to isolate the impact of the top-down hierarchy versus the flat LeWM baseline.
	* [ ] Design and feedback from others
	* [ ] Run and testing
	* [ ] Understand results (with the others)
* [ ] Test the efficacy of the additional regularization term on meta-actions to ensure no state information is "leaking" into the action representation.

#### Benchmarking & Analysis
* [ ] Execute systematic experiments to benchmark the hierarchy's reliability in medium and long-horizon tasks like PushT and Humanoid.
	* [ ] Define experiments and compare the results with the original paper (also visually)
* [ ] Profile the computational overhead to verify if planning remains efficient despite the added hierarchical levels.
	* [ ] Run experiments with some flags that track time or FLOPs (?)
* [ ] Perform hyperparameter tuning and sensitivity analysis, specifically on the temporal distances ($k_{1}, k_{2}$).
	* [ ] This should also tell which tasks are more long-range and which are short-range (if bigger $k$ improves results => long-range (?))
* [ ] Consolidate results into visual formats (graphs/tables) comparing performance against the LeWM baseline for the final report.
	* [ ] Better to automatize the process and save results in a JSON or CSV.
```
