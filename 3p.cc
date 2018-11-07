#include <string>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <random>
#include <math.h>
#include "avl.cc"

using namespace std;

#define MAX_PLAYERS 4

int NP = 4;
int ITERS = 10000000;

float vals[MAX_PLAYERS];
float range_vals[MAX_PLAYERS];
float rewards[MAX_PLAYERS];
float cumulative_rewards[MAX_PLAYERS];
float SPR = 1;
float learn_offset = 1;

struct Target {
	float value;
	float avg;
	int count;

	float lower;
	float upper;

	Target(float init, float lower, float upper) : 
		value(init), avg(init), count(0), lower(lower), upper(upper) {}

	void learn(float data) {
		value += data / sqrt(count + learn_offset);
		if (value > upper) value = upper;
		if (value < lower) value = lower;
		avg = (avg * count + value) / (count + learn_offset);
		count++;
	}
};

class History {
  public:
  	Target value_thresh;
  	Target bluff_thresh;

  	History() : value_thresh(Target(0, 0., .5)),  bluff_thresh(Target(1, .5, 1.)) {}

  	void learn_value(float regret) {
  		value_thresh.learn(regret);
  	}

  	void learn_bluff(float regret) {
  		bluff_thresh.learn(-regret);
  	}

  	float value() { return value_thresh.avg; }

  	float bluff() { return bluff_thresh.avg; }
};

struct GameState {
	string actions;
	float range_vals[MAX_PLAYERS];
	float range_lens[MAX_PLAYERS];

	int get_player() {
		return actions.length() % NP;
	}

	void take_action(char d, History& his) {
		int player = get_player();
		float rv = range_vals[player];
		float len1 = his.value() + 1.0 - his.bluff();
		float len0 = his.bluff() - his.value();
		actions += d;
		if (d == '1') {
			range_lens[player] *= len1;
			if (rv < his.value()) {
				range_vals[player] = rv / len1;
			} else if (rv < his.bluff()) {
				range_vals[player] = his.value() / len1;
			} else {
				range_vals[player] = (his.value() + rv) / len1;
			}
		} else {
			range_lens[player] *= len0;
			range_vals[player] = (rv - his.value()) / len0;
		}
	}

	bool showdown() {
		if (actions.length() >= NP && actions[get_player()] == '1') return true;
		else if (actions.length() == NP && actions == string(NP, '0')) return true;
		return false;
	}

	bool can_bluff() {
		return actions == string(actions.length(), '0');
	}
};

unordered_map<string, History*> model;

// Writes to rewards
void showdown(GameState& state) {
	bool has_aggressor = state.actions.length() > NP || state.actions[0] == '1';
	// cout << "has_aggressor:" << has_aggressor << endl;
	int winner = 0;
	float win_val = 2;
	float pot = 1;
	for (int i = 0; i < state.actions.length(); i++) {
		int p = i % NP;
		float cur_val = vals[p];
		char cur_state = state.actions[i];
		if (cur_val < win_val && (!has_aggressor || cur_state == '1')) {
			winner = p;
			win_val = cur_val;
		}
		if (cur_state == '1') {
			pot += SPR;
			rewards[p] = -SPR;
		} else {
			rewards[p] = 0;
		}
	}
	rewards[winner] += pot;
	// if (state.actions == "00111") {
	//	cout << "vals:" << vals[0] << ":" << vals[1] << ":" << vals[2] << endl;
	//	cout << "showdown:" << state.actions << ":" << rewards[0] << ":" << rewards[1] << ":" << rewards[2] << endl;
	//}
	/*
	cout << state << "--";
	for (int i = 0; i < NP; i++) {
		cout << rewards[i] << " ";
	}
	cout << endl;
	*/
}



void create_history(string actions) {
	auto it = model.find(actions);
	if (it == model.end()) {
		History* h = new History();
		model[actions] = h;
	}
}

GameState next_state(GameState& state) {
	GameState ret = state;
	int player = state.get_player();
	auto& his = *model[ret.actions];
	char decision;
	if (ret.range_vals[player] < his.value()) {
		ret.take_action('1', his);
	} else if (!ret.can_bluff() || ret.range_vals[player] < his.bluff()) {
		ret.take_action('0', his);
	} else {
		ret.take_action('1', his);
	}
	return ret;
}

// Writes to rewards
void get_value(GameState state) {
	while (!state.showdown()) {
		state = next_state(state);
	}
	showdown(state);
}

mt19937 rng(2822575);
uniform_real_distribution<> dis(0.0, 1.0);


void learn_spot(GameState& state) {

	// cout << "vals:" << vals[0] << ":" << vals[1] << endl;

	int p = state.get_player();

	History& his = *model[state.actions];
	float old_val = vals[p];

	GameState temp;

	vals[p] -= state.range_vals[p] * state.range_lens[p];
	vals[p] += his.value() * state.range_lens[p];

	float regret = 0;
	temp = state;
	temp.range_vals[p] = his.value();
	temp.take_action('1', his);
	get_value(temp);
	regret += rewards[p];
	temp = state;
	temp.range_vals[p] = his.value();
	temp.take_action('0', his);
	get_value(temp);
	regret -= rewards[p];

	// cout << "learning value:" << vals[p] << ":" << state.actions << ":" << regret << endl;

	his.learn_value(regret);

	if (state.can_bluff()) {

		vals[p] += (his.bluff() - his.value()) * state.range_lens[p];

		float regret = 0;
		temp = state;
		temp.range_vals[p] = his.bluff();
		temp.take_action('1', his);
		get_value(temp);
		// cout << "reward:" << rewards[p] << endl;
		regret += rewards[p];
		temp = state;
		temp.range_vals[p] = his.bluff();
		temp.take_action('0', his);
		get_value(temp);
		// cout << "reward:" << rewards[p] << endl;
		regret -= rewards[p];

		// cout << "learning bluff:" << vals[p] << ":" << state.actions << ":" << regret << endl;

		his.learn_bluff(regret);
	}

	vals[p] = old_val;
}

void simulate_game(bool prnt = false) {
	GameState state;
	state.actions = "";
	for (int i = 0; i < NP; i++) {
		vals[i] = dis(rng);
		state.range_vals[i] = vals[i];
		state.range_lens[i] = 1;
		// cout << vals[i] << " ";
	}
	// cout << endl;
	while (!state.showdown()) {
		GameState nstate = next_state(state);

		learn_spot(state);

		state = nstate;
	}

	showdown(state);
	for (int i = 0; i < NP; i++) {
		cumulative_rewards[i] += rewards[i];
	}
}

void initialize(GameState state, bool prnt = false) {
	if (state.showdown()) return;
	create_history(state.actions);
	History& his = *model[state.actions];
	if (prnt) {
		cout << state.actions << "\t" << his.value_thresh.avg;
		if (state.can_bluff()) cout << "\t" << (1.0 - his.bluff_thresh.avg);
		cout << endl;
	}
	GameState temp = state;
	state.take_action('0', his);
	initialize(state, prnt);
	temp.take_action('1', his);
	initialize(temp, prnt);
}

int main() {
	GameState state;
	state.actions = "";
	initialize(state);
	std::cout << std::fixed;
    std::cout << std::setprecision(4);

	for (int i = 1; i <= ITERS; ++i) {
		simulate_game();
		//if (i > 999900) {
		//	cout << model["00"]->value_thresh.value << endl;
		//}
	}
	initialize(state, true);

	cout << endl;

	for (int i = 0; i < NP; i++) {
		cout << "p" << i << " " << cumulative_rewards[i] / ITERS << endl;
	}

	// auto his = *model["0"];
	// his.t->printBalance();
	return 0;
}