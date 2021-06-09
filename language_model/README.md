# [DOC] Création d'un language model

## Prérequis

- kenlm

## Installation des librairies

---

### Dépendances

*Voir fichier BUILDING sur le git de kenlm pour l'installation sur d'autres OS*

```bash
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
```

### Compilation kenlm

```bash
mkdir -p build
cd build
cmake ..
make -j 4
```

## Création du corpus

---

Le corpus est créé à partir d'un dump wikipedia issu de commonvoice ainsi que d'un dump de débats à l'assemblée nationale.

```bash
./prepare_corpus.sh
```

Les sorties sont deux corpus : `sources_lm.txt` et `sources_lm_5.txt` qui est une version 5x plus petite de `sources_lm.txt`.

## Création du language model

```bash
kenlm/build/bin/lmplz -o 3 -S 80% --pruning 0 0 1 < sources_lm.txt > lm.arpa
```