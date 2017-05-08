#;load data first
import yaml
import json
import h5py
import numpy as np
from simple_dqn import encoder

def load_data(files):

    obj_list = []
    for file in files:
        with open(file) as f:
            for line in f:
                # print line
                try:
                    obj_list.append(yaml.safe_load(line))
                except:
                    print line
            # obj_list += [yaml.safe_load(line) for line in f]
                if len(obj_list) % 50 == 0: print len(obj_list)
    return obj_list

def test_load_file():

    files = ['sf_vs_sf_result.txt', 'sf_vs_sf_result_1.txt', 'sf_vs_sf_result_2.txt']
    for file_name in files:
        encode_feature(file_name)

def encode_feature(file_name):

    swap = [1, 0]

    minion_list = ['damagedgolem', 'clockworkgnome', 'boombot', 'manawyrm', 'cogmaster', 'annoyotron', 'mechwarper',
                   'snowchugger', 'harvestgolem', 'spidertank', 'tinkertowntechnician', 'mechanicalyeti',
                   'goblinblastmage', 'loatheb', 'archmageantonidas', 'drboom', 'unknown']

    non_minion_list = ['thecoin', 'armorplating', 'fireblast', 'frostbolt', 'fireball', 'flamestrike']

    card_list = minion_list + non_minion_list

    global_list = []
    own_board_list = []
    own_hand_list = []
    enemy_board_list = []
    enemy_hand_list = []
    own_future_list = []
    enemy_future_list = []
    playcard_list = []
    target_list = []
    # end_turn_ft_list = []
    own_deck_ft_list = []
    enemy_deck_ft_list = []
    sf_value_list = []
    random_list = []
    playable_list = []

    seq_len = []
    with open(file_name) as f:
        matchNo = 0
        num_state = 0
        for line in f:
            try:
                # print line
                obj = yaml.load(line)
            except:
                continue
            seq = obj['playSec']
            # if matchNo >= 1000: break

            # print matchNo, len(seq)
            seq_len.append(len(seq))
            matchNo += 1
            # continue

            skip = False
            if len(seq) > 30:
                continue

            result = obj['result']

            for j in xrange(len(seq)):

                # if (j % 2) == 0:
                #     actualResult = swap[result]
                # else:
                #     actualResult = result
                turn_start = seq[j]['turnSt']

                sf_val = seq[j]['stateValue']
                ownJson = seq[j]['attackPlayer']
                enemyJson = seq[j]['defensePlayer']

                actualResult = swap[result] if ownJson['turn'] == 1 else result

                is_random = 1 if seq[j]['isRandom'] else 0
                ownHeroHp = ownJson['heroInfo']['Hp']
                if turn_start:
                    ownMaxMana = ownJson['maxMana']
                ownMinionList = ownJson['minionJsonList']

                future_draw = []
                ownHandList = ownJson['handcardJsonList']

                # encode played hand
                non_minion_cost = 0
                own_played_cards = [0] * 10

                own_playedcard_list = ownJson['playedCardJsonList']
                # if own_playedcard_list and len(own_playedcard_list) != 0:
                #     for order, pc in enumerate(own_playedcard_list):
                #         # played_card = [0] * len(card_list)
                #         card_name = pc['cardName']
                #         if card_name in non_minion_list:
                #             non_minion_cost += pc['manacost']
                #         index = card_list.index(card_name)
                #         own_played_cards[order] = index

                enemyHeroHp = enemyJson['heroInfo']['Hp']
                if turn_start:
                    enemyMaxMana = enemyJson['maxMana']

                heroFeature = encoder.heroOneHot([ownMaxMana, ownHeroHp, enemyMaxMana, enemyHeroHp])

                # encode enemy minion
                enemyMinionList = enemyJson['minionJsonList']
                # enemy_board_feature = np.zeros((2,len(minion_list),5), dtype=int)

                enemy_dict = {}

                for m in enemyMinionList:
                    name = m['name']
                    eneity = m['entity']
                    enemy_dict[eneity] = name

                # #encode enemy hand
                enemyHandList = enemyJson['handcardJsonList']
                current_turn = j

                for k in range(5):
                    current_turn += 2
                    if current_turn < len(seq):
                        next_turn_json = seq[current_turn]['attackPlayer']
                        if len(next_turn_json['handcardJsonList']) != 0:
                            next_card_json = next_turn_json['handcardJsonList'][-1]
                            card_name = next_card_json['cardName']
                            future_draw.append(card_name)
                        else:
                            future_draw.append('None')
                    else:
                        future_draw.append('None')

                own_card_draw_feature = np.zeros((5, len(card_list)), dtype=int)
                for card_name in future_draw:
                    turn_idx = future_draw.index(card_name)
                    if card_name == 'None':
                        continue
                    card_index = card_list.index(card_name)
                    own_card_draw_feature[turn_idx, card_index] = 1

                future_draw = []

                current_turn = j+1
                for k in range(5):
                    card_draw = [0] * len(card_list)
                    current_turn += 2
                    if current_turn < len(seq):
                        next_turn_json = seq[current_turn]['attackPlayer']
                        if len(next_turn_json['handcardJsonList']) != 0:
                            next_card_json = next_turn_json['handcardJsonList'][-1]
                            card_name = next_card_json['cardName']
                            future_draw.append(card_name)
                        else:
                            future_draw.append('None')
                    else:
                        future_draw.append('None')

                enemy_card_draw_feature = np.zeros((5, len(card_list)), dtype=int)
                for card_name in future_draw:
                    turn_idx = future_draw.index(card_name)
                    if card_name == 'None':
                        continue
                    card_index = card_list.index(card_name)
                    enemy_card_draw_feature[turn_idx, card_index] = 1

                # encode target
                target_feature = [0] * len(minion_list)
                own_action_list = ownJson['playedActionJsonList']
                if own_action_list and len(own_action_list) != 0:
                    for action in own_action_list:
                        target_entity = action['targetEntity']
                        if target_entity in enemy_dict:
                            target_name = enemy_dict[target_entity]
                            index = minion_list.index(target_name)
                            target_feature[index] += 1
                            del enemy_dict[target_entity]

                own_deck_list = ownJson['ownDeckList']
                enemy_deck_list = enemyJson['ownDeckList']
                own_deck_feature, enemy_deck_feature = encoder.encode_deck(own_deck_list, enemy_deck_list)

                #adding playable
                playcard_action = [action for action in own_action_list if action['actionType'] == 1 or action['actionType'] == 3]
                canplay_heropower = ownJson['canPlayHeroPower']

                playable_change = ownJson['handcardChange']
                playable_change = [ownHandList] + playable_change if playable_change else [ownHandList]
                playable_feature = np.zeros((10, len(card_list)))
                use_hp = False
                for idx in xrange(len(playcard_action)):
                    actionPlayable = playable_change[idx]
                    action = playcard_action[idx]
                    if action['actionType'] == 3: use_hp = True
                    card_idx = idx - 1 if use_hp else idx
                    card_name = 'fireblast' if action['actionType'] == 3 else own_playedcard_list[card_idx]['cardName']

                    #update playable
                    for card in actionPlayable:
                        if card['playable']: playable_feature[idx, card_list.index(card['cardName'])] = 1
                    #heropower
                    if canplay_heropower[idx] == 1:
                        playable_feature[idx, card_list.index('fireblast')] = 1

                    # update played card
                    own_played_cards[idx] = card_list.index(card_name)

                # own_playable_feature = np.zeros(len(card_list))
                # for card in ownHandList:
                #     index = card_list.index(card['cardName'])
                #     if card['playable'] == 'true': own_playable_feature[index] = 1
                #
                # enemy_playable_feature = np.zeros(len(card_list))
                # for card in enemyHandList:
                #     index = card_list.index(card['cardName'])
                #     if card['playable'] == 'true': enemy_playable_feature[index] = 1
                #finish adding
                own_played_feature = [min(1.0, float(non_minion_cost) / ownMaxMana), non_minion_cost,
                                      ownMaxMana] + own_played_cards

                heroFeature, own_board_feature, own_hand_feature, enemy_board_feature, enemy_hand_feature = encoder.encode_board(
                    [ownMaxMana, ownHeroHp, enemyMaxMana, enemyHeroHp], ownHandList, ownMinionList, enemyHandList,
                    enemyMinionList)

                global_list.append([matchNo, j, actualResult] + heroFeature)
                own_board_list.append(own_board_feature)
                enemy_board_list.append(enemy_board_feature)
                own_hand_list.append(own_hand_feature)
                enemy_hand_list.append(enemy_hand_feature)
                own_future_list.append(own_card_draw_feature)
                enemy_future_list.append(enemy_card_draw_feature)

                random_list.append(is_random)
                playcard_list.append(own_played_feature)
                target_list.append(target_feature)
                # end_turn_ft_list.append(end_turn_ft)
                playable_list.append(playable_feature)
                # enemy_playable_list.append(enemy_playable_feature)

                own_deck_ft_list.append(own_deck_feature)
                enemy_deck_ft_list.append(enemy_deck_feature)
                sf_value_list.append(sf_val)

                # print 'hero:', heroFeature
                # if debug:
                #     print 'own_board_feature count:\n', own_board_feature[0,:,:]
                #     print 'own_board_feature special:\n', own_board_feature[1,:,:]
                #     pass

                # print 'enemy_board_feature:', enemy_board_feature
                # print 'own_hand_feature:', own_hand_feature
                # print 'enemy_hand_feature:', enemy_hand_feature
                # print 'own_card_draw_feature:', own_card_draw_feature
                # print 'enemy_card_draw_feature:', enemy_card_draw_feature
                #
                # print 'own:', ownMinionList
                # print 'enemy:', enemyMinionList

                num_state += 1
            if matchNo % 100 == 0:
                print matchNo

        # print matchNo
        # print '=========================================='
        # print sum(1 for dummy in seq_len if dummy > 120)
        # print sum(1 for dummy in seq_len if dummy > 130)
        # print sum(1 for dummy in seq_len if dummy > 140)
        # print sum(1 for dummy in seq_len if dummy > 150)
        # print sum(1 for dummy in seq_len if dummy > 160)
        # print sum(1 for dummy in seq_len if dummy > 170)
        # print sum(1 for dummy in seq_len if dummy > 180)
        # print sum(1 for dummy in seq_len if dummy > 190)
        # print sum(1 for dummy in seq_len if dummy > 200)


        f = h5py.File(file_name + "_ft.hdf5", "w")

        f.create_dataset('global', data=global_list)
        f.create_dataset('own_board', data=own_board_list)
        f.create_dataset('enemy_board', data=enemy_board_list)
        f.create_dataset('own_hand', data=own_hand_list)
        f.create_dataset('enemy_hand', data=enemy_hand_list)
        f.create_dataset('own_future', data=own_future_list)
        f.create_dataset('enemy_future', data=enemy_future_list)
        f.create_dataset('playcard_feature', data=playcard_list)
        f.create_dataset('target_feature', data=target_list)
        # f.create_dataset('endturn_feature', data=end_turn_ft_list)
        f.create_dataset('own_deck', data=own_deck_ft_list)
        f.create_dataset('enemy_deck', data=enemy_deck_ft_list)
        f.create_dataset('sf_value', data=sf_value_list)
        f.create_dataset('is_random', data=random_list)
        f.create_dataset('playable_feature', data=playable_list)
        f.close()

def encode_et_feature(file_name):

    swap = [1, 0]

    minion_list = ['damagedgolem', 'clockworkgnome', 'boombot', 'manawyrm', 'cogmaster', 'annoyotron', 'mechwarper',
                   'snowchugger', 'harvestgolem', 'spidertank', 'tinkertowntechnician', 'mechanicalyeti',
                   'goblinblastmage', 'loatheb', 'archmageantonidas', 'drboom', 'unknown']

    non_minion_list = ['thecoin', 'armorplating', 'fireblast', 'frostbolt', 'fireball', 'flamestrike']

    card_list = minion_list + non_minion_list

    target_list = []
    end_turn_ft_list = []

    with open(file_name) as f:
        matchNo = 0
        num_state = 0
        for line in f:
            try:
                # obj = yaml.safe_load(line)
                obj = json.loads(line)
            except:
                print line
                continue
            seq = obj['playSec']
            matchNo += 1
            if len(seq) > 30:
                continue

            if (len(seq) / 2) % 2 == 1:
                result = 1
            else:
                result = 0

            for j in xrange(len(seq)):

                if (j % 2) == 0:
                    actualResult = swap[result]
                else:
                    actualResult = result

                ownJson = seq[j]['attackPlayer']
                end_turn_ft = ownJson['endTurnFeatrueList']
                target_list.append([matchNo, j, actualResult])
                end_turn_ft_list.append(end_turn_ft)
                num_state += 1
            if matchNo % 100 == 0:
                print matchNo

        f = h5py.File(file_name + "_etft.hdf5", "w")

        f.create_dataset('global', data=target_list)
        f.create_dataset('endturn_feature', data=end_turn_ft_list)
        f.close()

        return end_turn_ft_list, target_list

def load_encoded_data(file_name):
    pass

if __name__ == '__main__':
    # test_load_file()
    # encode_feature('svs_result_1.txt')
    # encode_feature('svs_result_2.txt')
    # encode_feature('svs_result_3.txt')
    # encode_feature('ava_result_1.txt')
    # encode_feature('ava_result_2.txt')
    # encode_feature('ava_result_3.txt')
    encode_feature('svs_result_3.1.txt')
    # encode_feature('svs_result_2.1.txt')
    # encode_feature('svs_result_1.1.txt')