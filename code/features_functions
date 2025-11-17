from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import json

BASE_KEYS = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']

def build_species_index(*datasets):
    
    idx = {}

    def take(p):
        name = str(p.get('name', '')).lower()
        if not name: return
        rec = {k: int(p.get(k, 0)) for k in BASE_KEYS}
        if name not in idx or sum(idx[name].values()) < sum(rec.values()):
            idx[name] = rec

    for data in datasets:
        for b in data:
            for p in b.get('p1_team_details', []):
                take(p)
            if b.get('p2_lead_details'):
                take(b['p2_lead_details'])
            for ev in b.get('battle_timeline', []):
                take(ev.get('p2_pokemon_state', {}))
    return idx


species_idx = build_species_index(train_data)


def get_lead_speed(battle: dict, player: str, species_idx: dict) -> int:

    lead_name = None

    if player == 'p1':
        lead_name = battle.get('p1_lead_details', {}).get('name')
    elif player == 'p2':
        lead_name = battle.get('p2_lead_details', {}).get('name')

    if not lead_name:
        timeline = battle.get('battle_timeline', [])
        if timeline:
            if player == 'p1':
                lead_name = timeline[0].get('p1_pokemon_state', {}).get('name')
            else:
                lead_name = timeline[0].get('p2_pokemon_state', {}).get('name')

    if not lead_name and player == 'p1':
        team_p1 = battle.get('p1_team_details', [])
        if team_p1:
            lead_name = team_p1[0].get('name')

    if not lead_name:
        return 0

    stats = species_idx.get(str(lead_name).lower())
    if not stats:
        return 0

    return int(stats.get('base_spe', 0))


def track_pokemon_conditions(battle: dict):
  
    p1_pok_cond = {
        pokemon.get('name', f'p1_unknown_{i}'): {
            'hp': 1.0,
            'status': 'nostatus'
        }
        for i, pokemon in enumerate(battle.get('p1_team_details', []))
    }

    p2_pok_cond = {}
    p2_lead_name = battle.get('p2_lead_details', {}).get('name')
    if p2_lead_name:
        p2_pok_cond[p2_lead_name] = {
            'hp': 1.0,
            'status': 'nostatus'
        }

    timeline = battle.get('battle_timeline', []) or []

    for turn in timeline:
        p1_state = turn.get('p1_pokemon_state', {}) or {}
        p2_state = turn.get('p2_pokemon_state', {}) or {}

        name_p1 = p1_state.get('name')
        if name_p1:
            p1_pok_cond[name_p1] = {
                'hp': p1_state.get('hp_pct', 1.0),
                'status': p1_state.get('status', 'nostatus')
            }

        name_p2 = p2_state.get('name')
        if name_p2:
            p2_pok_cond[name_p2] = {
                'hp': p2_state.get('hp_pct', 1.0),
                'status': p2_state.get('status', 'nostatus')
            }

    p1_n_changes = 0
    p2_n_changes = 0
    p1_prev_name = None
    p2_prev_name = None

    for turn in timeline:
        p1_curr = turn.get('p1_pokemon_state', {}) or {}
        p2_curr = turn.get('p2_pokemon_state', {}) or {}

        name_curr_p1 = p1_curr.get('name')
        name_curr_p2 = p2_curr.get('name')

        if p1_prev_name is not None and name_curr_p1 != p1_prev_name:
            p1_n_changes += 1
        if p2_prev_name is not None and name_curr_p2 != p2_prev_name:
            p2_n_changes += 1

        p1_prev_name = name_curr_p1
        p2_prev_name = name_curr_p2

    if timeline:
        last_turn = timeline[-1]
        last_p1 = last_turn.get('p1_pokemon_state', {}) or {}
        last_p2 = last_turn.get('p2_pokemon_state', {}) or {}

        p1_effects = len(last_p1.get('effects', [])) * 0.5
        p2_effects = len(last_p2.get('effects', [])) * 0.5
    else:
        p1_effects = 0.0
        p2_effects = 0.0

    while len(p2_pok_cond) < 6:
        idx = len(p2_pok_cond)
        p2_pok_cond[f'p2_unknown_{idx}'] = {
            'hp': 1.0,
            'status': 'nostatus'
        }

    return p1_n_changes, p1_effects, p1_pok_cond, p2_n_changes, p2_effects, p2_pok_cond


def compute_differences_base_stats(p1_pok_cond: dict,
                                   p2_pok_cond: dict,
                                   species_idx: dict):
 
    p1_total_speed = p2_total_speed = 0
    p1_total_attack = p2_total_attack = 0
    p1_total_defense = p2_total_defense = 0
    p1_total_sp_attack = p2_total_sp_attack = 0
    p1_total_sp_defense = p2_total_sp_defense = 0
    p1_total_hp = p2_total_hp = 0

    for name in p1_pok_cond.keys():
        key = str(name).lower()
        if key in species_idx:
            p1_total_speed   += species_idx[key]['base_spe']
            p1_total_attack  += species_idx[key]['base_atk']
            p1_total_defense += species_idx[key]['base_def']
            p1_total_sp_attack   += species_idx[key]['base_spa']
            p1_total_sp_defense  += species_idx[key]['base_spd']
            p1_total_hp      += species_idx[key]['base_hp']

    for name in p2_pok_cond.keys():
        key = str(name).lower()
        if key in species_idx:
            p2_total_speed   += species_idx[key]['base_spe']
            p2_total_attack  += species_idx[key]['base_atk']
            p2_total_defense += species_idx[key]['base_def']
            p2_total_sp_attack   += species_idx[key]['base_spa']
            p2_total_sp_defense  += species_idx[key]['base_spd']
            p2_total_hp      += species_idx[key]['base_hp']

    speed      = p1_total_speed   - p2_total_speed
    defense    = p1_total_defense - p2_total_defense
    attack     = p1_total_attack  - p2_total_attack
    sp_attack  = p1_total_sp_attack  - p2_total_sp_attack
    sp_defense = p1_total_sp_defense - p2_total_sp_defense
    hp         = p1_total_hp      - p2_total_hp

    return speed, defense, attack, sp_attack, sp_defense, hp

def extract_boost_difference(timeline: list[dict]):

    boost_keys = ['atk', 'def', 'spa', 'spd', 'spe']

    prev_boosts_p1 = {}  
    prev_boosts_p2 = {}

    p1_total = 0
    p2_total = 0

    for ev in timeline:

        p1_state = ev.get("p1_pokemon_state", {}) or {}
        name_p1 = p1_state.get("name")

        if name_p1:
            curr_boosts_p1 = {
                k: p1_state.get("boosts", {}).get(k, 0)
                for k in boost_keys
            }

            prev_for_p1 = prev_boosts_p1.get(
                name_p1,
                {k: 0 for k in boost_keys}
            )

            for k in boost_keys:
                delta = curr_boosts_p1[k] - prev_for_p1[k]
                if delta > 0:
                    p1_total += delta

            prev_boosts_p1[name_p1] = curr_boosts_p1

        p2_state = ev.get("p2_pokemon_state", {}) or {}
        name_p2 = p2_state.get("name")

        if name_p2:
            curr_boosts_p2 = {
                k: p2_state.get("boosts", {}).get(k, 0)
                for k in boost_keys
            }

            prev_for_p2 = prev_boosts_p2.get(
                name_p2,
                {k: 0 for k in boost_keys}
            )

            for k in boost_keys:
                delta = curr_boosts_p2[k] - prev_for_p2[k]
                if delta > 0:
                    p2_total += delta

            prev_boosts_p2[name_p2] = curr_boosts_p2

    return p1_total, p2_total, p1_total - p2_total

def extract_accuracy_difference(timeline: list[dict]):
    
    p1_acc_values = []
    p2_acc_values = []

    for ev in timeline:
       
        p1_move = ev.get("p1_move_details") or {}
        if "accuracy" in p1_move:
            acc = p1_move.get("accuracy", None)
            if isinstance(acc, (int, float)):
                p1_acc_values.append(acc)

        p2_move = ev.get("p2_move_details") or {}
        if "accuracy" in p2_move:
            acc = p2_move.get("accuracy", None)
            if isinstance(acc, (int, float)):
                p2_acc_values.append(acc)

    p1_mean = np.mean(p1_acc_values) if p1_acc_values else 0.0
    p2_mean = np.mean(p2_acc_values) if p2_acc_values else 0.0

    return p1_mean, p2_mean, p1_mean - p2_mean
