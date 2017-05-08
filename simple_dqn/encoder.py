import json
import numpy as np
from sklearn.externals import joblib
import h5py

swap = [1, 0]
heroHeader = ["ownHp", "ownMaxMana"]

minion_list = ["damagedgolem", "clockworkgnome", "boombot", "manawyrm", "cogmaster", "annoyotron", "mechwarper",
               "snowchugger", "harvestgolem", "spidertank", "tinkertowntechnician", "mechanicalyeti",
               "goblinblastmage", "loatheb", "archmageantonidas", "drboom", "unknown"]

non_minion_list = ["thecoin", "armorplating", "fireblast", "frostbolt", "fireball", "flamestrike"]

card_list = minion_list + non_minion_list

minionVal = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 17.0, 30.0]
mana_enc = joblib.load('C:\Code\RL_for_hs\simple_dqn\encoder\mana.pkl')
hp_enc = joblib.load('C:\Code\RL_for_hs\simple_dqn\encoder\hp.pkl')

class FeatureEncoder:

    def __init__(self):
        self.global_ft = []
        self.board_ft = []
        self.hand_ft = []
        self.play_ft = []
        self.target = []
        self.global_shape = (36,)
        self.board_shape = (9, 17, 5,)
        self.hand_shape = (9, 23,)
        self.play_shape = (23,)
        self.target_shape = (40,)

    def fill_ft_str(self, msg):

        def process(chunk):
            if not chunk: return None
            return map(lambda x: map(int, x.split(',')), chunk.split('.'))

        idx = map(process, msg.split('|'))
        global_feature = np.zeros(self.global_shape, dtype=np.float32)
        board_feature = np.zeros(self.board_shape, dtype=np.float32)
        hand_feature = np.zeros(self.hand_shape, dtype=np.float32)
        play_feature = np.zeros(self.play_shape, dtype=np.float32)

        if idx[0]: global_feature[idx[0]] = 1.0
        if idx[1]: board_feature[idx[1]] = 1.0
        if idx[2]: hand_feature[idx[2]] = 1.0
        if idx[3]: play_feature[idx[3]] = 1.0

        self.global_ft.append(global_feature)
        self.board_ft.append(board_feature)
        self.hand_ft.append(hand_feature)
        self.play_ft.append(play_feature)

    def fill_board(self, idx):
        ft = np.zeros(self.board_shape, dtype=np.float32)
        ft[idx] = 1.0
        self.board_ft.append(ft)

    def fill_hand(self, idx):
        ft = np.zeros(self.hand_shape, dtype=np.float32)
        ft[idx] = 1.0
        self.hand_ft.append(ft)

    def fill_global(self, idx):
        ft = np.zeros(self.global_shape, dtype=np.float32)
        ft[idx] = 1.0
        self.global_ft.append(ft)

    def fill_play(self, idx):
        ft = np.zeros(self.play_shape, dtype=np.float32)
        ft[idx] = 1.0
        self.play_ft.append(ft)

    def fill_target(self, idx):
        ft = np.zeros(self.target_shape, dtype=np.float32)
        ft[idx] = 1.0
        self.target.append(ft)

    def fill_target_str(self, str):

        def process(chunk):
            return map(lambda x: map(int, x.split(',')), chunk.split('.'))

        idx = process(str)
        ft = np.zeros(self.target_shape, dtype=np.float32)
        ft[idx] = 1.0
        self.target.append(ft)

    def write_h5(self, file_name):
        outFile = h5py.File(file_name, "w")
        outFile.create_dataset("globalList",data=np.array(self.global_ft, dtype=np.float32))
        outFile.create_dataset("boardList",data=np.array(self.board_ft, dtype=np.float32))
        outFile.create_dataset("handList", data=np.array(self.hand_ft, dtype=np.float32))
        outFile.create_dataset("playList", data=np.array(self.play_ft, dtype=np.float32))
        outFile.create_dataset("target", data=np.array(self.target, dtype=np.float32))
        outFile.close()

def heroOneHot(heroFeature):

    ownMaxMana = heroFeature[0]
    own_mana_feature = np.zeros(10).astype(int).tolist()
    own_mana_feature[ownMaxMana - 1] = 1

    ownHeroHp = heroFeature[1]
    own_hp_feature = np.select(
                      [ ownHeroHp <= 6,
                        6 < ownHeroHp <= 12,
                        12 < ownHeroHp <= 15,
                        15 < ownHeroHp <= 20,
                        20 < ownHeroHp],
                      [[1, 0, 0 ,0 ,0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1]]).astype(int).tolist()

    enemyMaxMana = heroFeature[2]
    enemy_mana_feature = np.zeros(10).astype(int).tolist()
    enemy_mana_feature[min(9,enemyMaxMana)] = 1

    enemyHeroHp = heroFeature[3]
    enemy_hp_feature = np.select(
                      [ enemyHeroHp <= 6,
                        6 < enemyHeroHp <= 12,
                        12 < enemyHeroHp <= 15,
                        15 < enemyHeroHp <= 20,
                        20 < enemyHeroHp],
                      [[1, 0, 0 ,0 ,0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1]]).astype(int).tolist()

    return own_mana_feature + own_hp_feature + enemy_mana_feature + enemy_hp_feature

def get_minion_special(name_idx):
    name = minion_list[name_idx]
    if name == "mechwarper":
        return 3
    if name in ["archmageantonidas", "manawyrm"]:
        return 2
    if name in ["clockworkgnome", "mechanicalyeti", "harvestgolem", "annoyotron", "boombot"]:
        return 1
    return 0

def minionOneHot(hp, attack, name_idx):

    hp_idx = np.select(
        [hp <= 1,
         1 < hp <= 2,
         2 < hp <= 4,
         4 < hp <= 6,
         6 < hp],
        [0, 1, 2, 3, 4]).astype(int).tolist()

    special = get_minion_special(name_idx)

    idx = [name_idx, hp_idx], special
    return idx

# def displayDictHist(dict):
#     X = np.arange(len(dict))
#     plt.bar(X, dict.values(), align="center", width=0.5)
#     plt.xticks(X, dict.keys())
#     ymax = max(dict.values()) + 1
#     plt.ylim(0, ymax)
#     plt.show()

def list_to_bitset(m_list):
    bitset = set()
    for i in range(len(m_list)):
        if m_list[i] == 1:
            bitset.add(i)

    return bitset

def encode_deck(own_deck_list, enemy_deck_list):

    own_deck_feature = np.zeros(len(card_list))
    for card_name in own_deck_list:
        index = card_list.index(card_name)
        own_deck_feature[index] += 1

    enemy_deck_feature = np.zeros(len(card_list))
    for card_name in enemy_deck_list:
        index = card_list.index(card_name)
        enemy_deck_feature[index] += 1

    return own_deck_feature, enemy_deck_feature

def encode_board(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list):
    hero_feature = heroOneHot(hero)

    own_hand_feature =  np.zeros(len(card_list))
    for hc in own_hand_list:
        card_name = hc["cardName"]
        index = card_list.index(card_name)
        own_hand_feature[index] += 1

    enemy_hand_feature =  np.zeros(len(card_list))
    for hc in enemy_hand_list:
        card_name = hc["cardName"]
        index = card_list.index(card_name)
        enemy_hand_feature[index] += 1

    own_board_feature = np.zeros((2, len(minion_list), 5), dtype=int)
    for m in own_minion_list:
        hp = m["Hp"]
        attack = m["Angr"]
        name = m["name"]
        nameIdx = minion_list.index(name)
        idx, special = minionOneHot(hp, attack, nameIdx)
        own_board_feature[0, idx[0], idx[1]] += 1
        own_board_feature[1, idx[0], idx[1]] = special

    enemy_board_feature = np.zeros((2, len(minion_list), 5), dtype=int)
    for m in enemy_minion_list:
        hp = m["Hp"]
        attack = m["Angr"]
        name = m["name"]
        nameIdx = minion_list.index(name)
        idx, special = minionOneHot(hp, attack, nameIdx)
        enemy_board_feature[0, idx[0], idx[1]] += 1
        enemy_board_feature[1, idx[0], idx[1]] = special

    return hero_feature, own_board_feature, own_hand_feature, enemy_board_feature, enemy_hand_feature

def encode_hand(own_hand_list, enemy_hand_list):

    own_hand_feature = np.zeros(len(card_list))
    for hc in own_hand_list:
        card_name = hc["cardName"]
        index = card_list.index(card_name)
        own_hand_feature[index] += 1

    enemy_hand_feature = np.zeros(len(card_list))
    for hc in enemy_hand_list:
        card_name = hc["cardName"]
        index = card_list.index(card_name)
        enemy_hand_feature[index] += 1

    return own_hand_feature, enemy_hand_feature

def encode_for_cnn_phase(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list, own_deck, enemy_deck):
    hero_feature, own_board_feature, own_hand_feature, enemy_board_feature, enemy_hand_feature = encode_board(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list)
    own_deck_feature, enemy_deck_feature = encode_deck(own_deck, enemy_deck)
    deck_feature = one_hot_hand_s(own_deck_feature, enemy_deck_feature, flatten=False)
    global_feature = one_hot_hero_phase_s(hero_feature)
    board_feature = one_hot_board_s(own_board_feature, enemy_board_feature, flatten=False)
    hand_feature = one_hot_hand_s(own_hand_feature, enemy_hand_feature, flatten=False)
    handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)
    return global_feature[:,0], [global_feature[:,1:], board_feature, handdeck_feature]

def encode_for_cnn_phase(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list, own_deck, enemy_deck, played_list, playable_list):
    hero_feature, own_board_feature, own_hand_feature, enemy_board_feature, enemy_hand_feature = encode_board(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list)
    own_deck_feature, enemy_deck_feature = encode_deck(own_deck, enemy_deck)
    deck_feature = one_hot_hand_s(own_deck_feature, enemy_deck_feature, flatten=False)
    global_feature = one_hot_hero_phase_s(hero_feature)
    board_feature = one_hot_board_s(own_board_feature, enemy_board_feature, flatten=False)
    hand_feature = one_hot_hand_s(own_hand_feature, enemy_hand_feature, flatten=False)
    handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)
    played_feature = one_hot_played_s(played_list)
    playable_feature = one_hot_playable(playable_list)
    play_feature = np.concatenate([playable_feature, played_feature], axis=1)
    return global_feature[:,0], [global_feature[:,1:], board_feature, handdeck_feature, play_feature]

def encodeSA_for_cnn_phase(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list, own_deck, enemy_deck, playcard_feature):
    hero_feature, own_board_feature, own_hand_feature, enemy_board_feature, enemy_hand_feature = encode_board(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list)
    own_deck_feature, enemy_deck_feature = encode_deck(own_deck, enemy_deck)
    deck_feature = one_hot_hand_s(own_deck_feature, enemy_deck_feature, flatten=False)
    global_feature = one_hot_hero_phase_s(hero_feature)
    board_feature = one_hot_board_s(own_board_feature, enemy_board_feature, flatten=False)
    hand_feature = one_hot_hand_s(own_hand_feature, enemy_hand_feature, flatten=False)
    handdeck_feature = np.concatenate([hand_feature, deck_feature], axis=2)
    return global_feature[:,0], [global_feature[:,1:], board_feature, handdeck_feature, playcard_feature]

def encode_for_cnn(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list):
    hero_feature, own_board_feature, own_hand_feature, enemy_board_feature, enemy_hand_feature = encode_board(hero, own_hand_list, own_minion_list, enemy_hand_list, enemy_minion_list)
    global_feature = one_hot_hero_s(hero_feature)
    board_feature = one_hot_board_s(own_board_feature, enemy_board_feature, flatten=False)
    hand_feature = one_hot_hand_s(own_hand_feature, enemy_hand_feature, flatten=False)
    return [global_feature, board_feature, hand_feature]

def encode_playcard(own_playedcard_list):
    own_played_feature = [0] * len(card_list)

    if len(own_playedcard_list) != 0:
        for pc in own_playedcard_list:
            card_name = pc['cardName']
            index = card_list.index(card_name)
            own_played_feature[index] = 1

    return own_played_feature

def encode_target(target_name_list):
    target_feature = [0] * len(minion_list)
    for target_name in target_name_list:
        index = minion_list.index(target_name)
        target_feature[index] = 1

    return target_feature

def encode_playtarget_feature(own_playedcard_list, target_name_list):
    res = encode_playcard(own_playedcard_list) + encode_target(target_name_list)

def one_hot_hero_phase_s(global_feature):

    own_mana = global_feature[:10]
    own_hp = global_feature[10:15]
    enemy_mana = global_feature[15:25]
    enemy_hp = global_feature[25:]
    mana_factor = np.arange(10).reshape(10, 1)
    own_mana_feature = np.dot(own_mana, mana_factor)
    enemy_mana_feature = np.dot(enemy_mana, mana_factor)
    hp_factor = np.arange(5).reshape(5, 1)
    own_hp_feature = np.dot(own_hp, hp_factor)
    enemy_hp_feature = np.dot(enemy_hp, hp_factor)
    hp_feature = own_hp_feature * 10 + enemy_hp_feature
    transformed_hp = hp_enc.transform(hp_feature.reshape(-1, 1)).toarray().flatten()
    global_feature = np.concatenate([own_mana_feature, enemy_mana_feature - own_mana_feature, transformed_hp], axis=0)
    return global_feature[None, :]

def one_hot_hero_phase(global_feature):
    own_mana = global_feature[:, :10]
    own_hp = global_feature[:, 10:15]
    enemy_mana = global_feature[:, 15:25]
    enemy_hp = global_feature[:, 25:]

    mana_factor = np.arange(10).reshape(10, 1)
    own_mana_feature = np.dot(own_mana, mana_factor)
    enemy_mana_feature = np.dot(enemy_mana, mana_factor)

    hp_factor = np.arange(5).reshape(5, 1)
    own_hp_feature = np.dot(own_hp, hp_factor)
    enemy_hp_feature = np.dot(enemy_hp, hp_factor)

    hp_feature = own_hp_feature * 10 + enemy_hp_feature
    transformed_hp = hp_enc.transform(hp_feature).toarray()

    global_feature = np.concatenate([own_mana_feature, enemy_mana_feature - own_mana_feature, transformed_hp], axis=1)

    return global_feature

def one_hot_hero(global_feature):
    own_mana = global_feature[:,:10]
    own_hp = global_feature[:,10:15]
    enemy_mana = global_feature[:,15:25]
    enemy_hp = global_feature[:,25:]

    mana_factor = np.arange(10).reshape(10, 1)
    own_mana_feature =  np.dot(own_mana, mana_factor)
    enemy_mana_feature = np.dot(enemy_mana, mana_factor)

    mana_feature = own_mana_feature * 10 + enemy_mana_feature

    hp_factor = np.arange(5).reshape(5, 1)
    own_hp_feature = np.dot(own_hp, hp_factor)
    enemy_hp_feature = np.dot(enemy_hp, hp_factor)

    hp_feature = own_hp_feature * 10 + enemy_hp_feature

    # enc = OneHotEncoder()
    # enc.fit(mana_feature)
    # joblib.dump(enc, 'encoder/hp.pkl')
    transformed_mana = mana_enc.transform(mana_feature).toarray()

    # enc = OneHotEncoder()
    # enc.fit(hp_feature)
    # joblib.dump(enc, 'encoder/mana.pkl')
    transformed_hp = hp_enc.transform(hp_feature).toarray()

    global_feature = np.concatenate([transformed_hp, transformed_mana], axis=1)

    return global_feature

def one_hot_hero_s(global_feature):
    own_mana = global_feature[:10]
    own_hp = global_feature[10:15]
    enemy_mana = global_feature[15:25]
    enemy_hp = global_feature[25:]

    mana_factor = np.arange(10).reshape(10, 1)
    own_mana_feature =  np.dot(own_mana, mana_factor)
    enemy_mana_feature = np.dot(enemy_mana, mana_factor)

    mana_feature = own_mana_feature * 10 + enemy_mana_feature

    hp_factor = np.arange(5).reshape(5, 1)
    own_hp_feature = np.dot(own_hp, hp_factor)
    enemy_hp_feature = np.dot(enemy_hp, hp_factor)

    hp_feature = own_hp_feature * 10 + enemy_hp_feature

    transformed_mana = mana_enc.transform(mana_feature.reshape(-1, 1)).toarray().flatten()

    transformed_hp = hp_enc.transform(hp_feature.reshape(-1, 1)).toarray().flatten()

    global_feature = np.concatenate([transformed_hp, transformed_mana], axis=0)

    return global_feature[None, :]

def one_hot_board_s(own_board, enemy_board, flatten=False):
    board_shape = [17, 5]

    own_board_count = own_board[0, :, :]
    enemy_board_count = enemy_board[0, :, :]
    count = own_board_count * 10 + enemy_board_count
    count = count.reshape(-1, 1)

    count = np.select(
        [count == 2,
         count == 1,
         count == 12,
         count == 0,
         count == 11,
         count == 22,
         count == 21,
         count == 10,
         count == 20],
        [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
         ])

    own_board_special = own_board[1, :, :]
    enemy_board_special = enemy_board[1, :, :]
    special = own_board_special * 10 + enemy_board_special
    special = special.reshape(-1, 1)

    special = np.select(
        [special == 3,
         special == 2,
         special == 1,
         special == 33,
         special == 22,
         special == 11,
         special == 0,
         special == 10,
         special == 20,
         special == 30],
        [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
         ])

    one_hot_length = 9
    transformed_count = count.reshape(board_shape[0], board_shape[1], one_hot_length)

    one_hot_length = 10
    transformed_special = special.reshape(board_shape[0], board_shape[1], one_hot_length)

    board_feature = np.concatenate([transformed_count, transformed_special], axis=2)
    board_feature = np.rollaxis(board_feature, 2, 0)

    if flatten:
        board_feature = board_feature.reshape(19 * 17 * 5)

    return board_feature[None, :]

def one_hot_board(own_board, enemy_board, flatten=False):

    board_shape = [17, 5]

    own_board_count = own_board[:,0,:,:]
    enemy_board_count = enemy_board[:,0,:,:]
    count = own_board_count * 10 + enemy_board_count
    count = count.reshape(-1, 1)

    count = np.select(
        [count == 2,
         count == 1,
         count == 12,
         count == 0,
         count == 11,
         count == 22,
         count == 21,
         count == 10,
         count == 20],
        [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        ])

    own_board_special = own_board[:,1,:,:]
    enemy_board_special = enemy_board[:,1,:,:]
    special = own_board_special * 10 + enemy_board_special
    special = special.reshape(-1, 1)

    special = np.select(
        [special == 3,
         special == 2,
         special == 1,
         special == 33,
         special == 22,
         special == 11,
         special == 0,
         special == 10,
         special == 20,
         special == 30],
        [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
         ])

    one_hot_length = 9
    transformed_count = count.reshape(-1, board_shape[0], board_shape[1], one_hot_length)

    one_hot_length = 10
    transformed_special = special.reshape(-1, board_shape[0], board_shape[1], one_hot_length)

    board_feature = np.concatenate([transformed_count, transformed_special], axis=3)
    board_feature = np.rollaxis(board_feature, 3, 1)

    if flatten:
        board_feature = board_feature.reshape(-1, 19 * 17 * 5)

    return board_feature

def one_hot_hand_s(own_hand, enemy_hand, flatten = False):
    hand_length = 23
    hand = np.clip(own_hand, 0, 2) * 10 + np.clip(enemy_hand, 0, 2)
    hand = hand.reshape(-1, 1)

    # print (np.unique(hand))

    hand = np.select(
        [hand == 2,
         hand == 1,
         hand == 12,
         hand == 0,
         hand == 11,
         hand == 22,
         hand == 21,
         hand == 10,
         hand == 20],
        [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
         ])

    one_hot_length = 9
    hand_feature = hand.reshape(hand_length, one_hot_length)

    hand_feature = np.rollaxis(hand_feature, 1, 0)

    if flatten:
        hand_feature = hand_feature.reshape(9 * 23)

    return hand_feature[None, :]

def one_hot_played_s(played_list):
    played_feature = np.zeros(len(card_list))
    for card_name in played_list:
        index = card_list.index(card_name)
        played_feature[index] += 1
    return played_feature[None, :]

def one_hot_playable(playable_list):
    played_feature = np.zeros(len(card_list))
    for card_name in playable_list:
        index = card_list.index(card_name)
        played_feature[index] = 1
    return played_feature[None, :]

def one_hot_hand(own_hand, enemy_hand, flatten = False):
    hand_length = 23
    hand = np.clip(own_hand, 0, 2) * 10 + np.clip(enemy_hand, 0, 2)
    hand = hand.reshape(-1, 1)

    # print (np.unique(hand))

    hand = np.select(
        [hand == 2,
         hand == 1,
         hand == 12,
         hand == 0,
         hand == 11,
         hand == 22,
         hand == 21,
         hand == 10,
         hand == 20],
        [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
         ])

    one_hot_length = 9
    hand_feature = hand.reshape(-1, hand_length, one_hot_length)

    hand_feature = np.rollaxis(hand_feature, 2, 1)

    if flatten:
        hand_feature = hand_feature.reshape(-1, 9 * 23)

    return hand_feature

def one_hot_future(own_future, enemy_future, flatten = False):
    hand_length = 23
    own_future = np.sum(own_future, axis=1)
    enemy_future = np.sum(enemy_future, axis=1)

    # print(own_future.shape)
    # print(enemy_future.shape)
    # print (own_future[:5])
    # print (enemy_future[:5])

    hand = np.clip(own_future, 0, 2) * 10 + np.clip(enemy_future, 0, 2)
    # enc = OneHotEncoder()
    hand = hand.reshape(-1, 1)

    hand = np.select(
        [hand == 2,
         hand == 1,
         hand == 12,
         hand == 0,
         hand == 11,
         hand == 22,
         hand == 21,
         hand == 10,
         hand == 20],
        [np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
         np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
         ])
    # print(np.unique(hand))

    one_hot_length = 9

    hand_future = hand.reshape(-1, hand_length, one_hot_length)
    hand_future = np.rollaxis(hand_future, 2, 1)

    if flatten:
        hand_future = hand_future.reshape(-1, 9 * 23)

    return hand_future

def train_test_split(feature_list, result_list, percent = 0.8, random_state = 42):

    rng = np.random.RandomState(random_state)

    assert len(feature_list[0]) == len(result_list)

    mask = rng.choice([False, True], len(result_list), p=[1.0-percent, percent])
    new_feature = []
    for feature in feature_list:
        new_data = feature[mask]
        new_feature.append(new_data)
    new_result = result_list[mask]

    mask = rng.choice([False, True], len(new_result), p=[0.2, 0.8])
    reverse_mask = np.array([False if m else True for m in mask])

    new_train_list = []
    new_test_list = []
    for feature in new_feature:
        test_data = feature[reverse_mask]
        new_data = feature[mask]
        new_train_list.append(new_data)
        new_test_list.append(test_data)

    test_result = new_result[reverse_mask]
    new_result = new_result[mask]

    return new_train_list, new_result, new_test_list, test_result

def test():
    pass

if __name__ == '__main__':
    test()
