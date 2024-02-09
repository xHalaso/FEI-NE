import math
import numpy as np
import random


# ------------------------------ VYBER ------------------------------

# elitaristicky vyber
# funkcia zoradi jedince podla uspesnoti a vytvori
# novu populaciu podla n_listu
def selbest(old_pop, old_pop_fit, n_list, reverse=False):

    new_pop, new_fit = [], []

    # vytvorenie pola indexov zoradenych jedincov
    fit_index = sorted(range(len(old_pop_fit)),
                       key=lambda k: old_pop_fit[k], reverse=reverse)

    # vytvorenie novej populacie podla n_listu
    for i in range(len(n_list)):
        for j in range(n_list[i]):
            new_pop.append(old_pop[fit_index[i]])
            new_fit.append(old_pop_fit[fit_index[i]])

    return np.array(new_pop), np.array(new_fit)


# elitaristicky vyber so zoradenim
# funkcia zoradi jedince a vrati prvych n jedincov
# v novej pop
def selsort(old_pop, old_pop_fit, n, reverse=False):

    new_pop, new_fit = [], []

    # vytvorenie listu indexov so zoradenymi jedincami
    fit_index = sorted(range(len(old_pop_fit)),
                       key=lambda k: old_pop_fit[k], reverse=reverse)

    # vytvorenie novej populacie s n jedincami
    for i in range(n):
        new_pop.append(old_pop[fit_index[i]])
        new_fit.append(old_pop_fit[fit_index[i]])

    return np.array(new_pop), np.array(new_fit)


# vyber na zaklade diverzity
# funkcia zisti diverzity jedincov od referencneho jedinca
# a podla n_listu vytvori novu pop
def seldiv(old_pop, old_pop_fit, n_list, mode, reverse=False):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape
    new_pop, new_fit = [], []

    # urcenie referencneho jedinca
    if mode == 1:
        reference_list = old_pop[sorted(
            range(len(old_pop_fit)), key=lambda k: old_pop_fit[k], reverse=reverse)[0]]
    elif mode == 0:
        reference_list = np.mean(old_pop, axis=0)

    # vytvorenie pola rozdielov jedincov od referenceho jedinca
    diversity_list = np.zeros([shape[0]])
    div_index = 0
    for i in range(shape[0]):
        diversity_list[i] = np.sum(np.abs(reference_list - old_pop[i]))

    # vytvorenie novej populacie podla n_listu
    for i in range(len(n_list)):
        maxi, div_index = 0, 0

        for j in range(shape[0]):
            if (diversity_list[j] > maxi) and (old_pop_fit[j] not in new_fit):
                maxi = diversity_list[j]
                div_index = j

        diversity_list[div_index] = 0

        for j in range(n_list[i]):
            new_pop.append(old_pop[div_index])
            new_fit.append(old_pop_fit[div_index])

    return np.array(new_pop), np.array(new_fit)


# nahodny vyber
# funkcia vytvori novu pop z nahodnych jedincov
def selrand(old_pop, old_pop_fit, n):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape
    new_pop, new_fit = [], []

    # vytvorenie novej populacie s n jedincami
    for i in range(n):
        j = math.ceil(random.random() * shape[0]) - 1
        new_pop.append(old_pop[j])
        new_fit.append(old_pop_fit[j])

    return np.array(new_pop), np.array(new_fit)


# vyber vahovanym ruletovym kolesom
# funkcia rozdeli rozsah 0 az 100 na vyseky, nasledne
# vyberie n cisel a podla toho do ktoreho vyseku
# patria bude jedinec ktoremu vysek patri zaradeny
# do novej pop, vyber cisel je vypocitany z poctu cisel n
# a nahodneho cisla
def selsus(old_pop, old_pop_fit, n, reverse=False):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape
    new_pop, new_fit = [], []

    # vypocet vysekov pre jedince podla ich uspesnoti
    if reverse:
        weights_help = np.append(
            1 / ((abs(np.subtract(old_pop_fit, max(old_pop_fit))) + 1) * sum(old_pop_fit)), 0)
    else:
        weights_help = np.append(
            1 / ((np.subtract(old_pop_fit, min(old_pop_fit)) + 1) * sum(old_pop_fit)), 0)

    weights = np.zeros(shape[0] + 1)
    for i in range(shape[0] - 1, -2, -1):
        weights[i] = weights[i + 1] + weights_help[i]
    weights[-1] = 0

    max_weight = max(weights)
    if max_weight == 0:
        max_weight = 0.00001

    weights /= (max_weight / 100)

    # vyber cisel
    roulette_help = random.uniform(0, 1) * (100 / n) - 0.00001
    roulette = np.array([(i * (100 / n) + roulette_help) for i in range(n)])

    # hladanie cisel vo vysekoch a skopirovanie jedinca do novej pop
    for i in range(n):
        for j in range(shape[0]):
            if (roulette[i] <= weights[j]) and (roulette[i] >= weights[j + 1]):
                break

        new_pop.append(old_pop[j])
        new_fit.append(old_pop_fit[j])

    return np.array(new_pop), np.array(new_fit)


# turnajovy vyber
# funkcia vykona "zapasy" medzi nahodne vybranymi jedincami
# lepsi jedinec bude skopirovany do novej pop
def seltourn(old_pop, old_pop_fit, n, reverse=False):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape
    new_pop, new_pop_fit = [], []

    # hlavny cyklus
    for i in range(n):

        # vyber dvojic
        j = math.ceil(shape[0] * random.random()) - 1
        k = math.ceil(shape[0] * random.random()) - 1

        # ak sa vyberie ten isty
        if j == k:
            new_pop.append(old_pop[j])
            new_pop_fit.append(old_pop_fit[j])

        # ak ma prvy lepsiu fit ako druhy a reverse==False
        elif old_pop_fit[j] <= old_pop_fit[k] and not reverse:
            new_pop.append(old_pop[j])
            new_pop_fit.append(old_pop_fit[j])

        # ak ma prvy lepsiu fit ako druhy a reverse==True
        elif old_pop_fit[j] <= old_pop_fit[k] and reverse:
            new_pop.append(old_pop[k])
            new_pop_fit.append(old_pop_fit[k])

        # ak ma druhy lepsiu fit ako prvy
        else:
            new_pop.append(old_pop[k])
            new_pop_fit.append(old_pop_fit[k])

    return np.array(new_pop), np.array(new_pop_fit)


# ------------------------------ MUTACIE ------------------------------

# obycajna mutacia
# funckia zmeni nahodne vybrane geny na cisla
# v rozsahu danom parametrom space
# intenzita mutacie je v parametri rate
def mutx(old_pop, rate, space):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape

    # kontrola ci faktor mutacie je v rozsahu 0 az 1
    rate = rate_check(rate)

    # vypocet poctu mutacii
    muts = math.floor(shape[0] * shape[1] * rate)

    # rozdiel v ohraniceniach (pomocne pre dalsie vypocty)
    space_dif = np.subtract(space[1], space[0])

    # cyklus vyberu genu a jeho mutacia
    for i in range(muts):
        row = math.ceil(random.random() * shape[0]) - 1
        column = math.ceil(random.random() * shape[1]) - 1

        # mutacia cez pomocnu funkciu, ktora kontroluje ci je nova hodnota v rozsahu space
        old_pop[row, column] = mut_check(random.random(
        ) * space_dif[column] + space[0, column], space[0, column], space[1, column])


# aditivna mutacia
# funckia zmeni nahodne vybrane geny na cisla tak ze
# ku genom bude pripocitane cislo ktoreho rozsahy
# su dane absolutnou hodnotou v matici amp
# intenzita mutacie je v parametri rate
def muta(old_pop, rate, amp, space):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape

    # kontrola ci faktor mutacie je v rozsahu 0 az 1
    rate = rate_check(rate)

    # vypocet poctu mutacii
    muts = math.floor(shape[0] * shape[1] * rate)

    # cyklus vyberu genu a jeho mutacia
    for i in range(muts):
        row = math.ceil(random.random() * shape[0]) - 1
        column = math.ceil(random.random() * shape[1]) - 1

        # mutacia cez pomocnu funkciu, ktora kontroluje ci je nova hodnota v rozsahu space
        old_pop[row, column] = mut_check(
            old_pop[row, column] + random.uniform(-1, 1) * amp[column], space[0, column], space[1, column])


# multiplikativna mutacie
# funckia zmeni nahodne vybrane geny na cisla tak ze
# gen budu vynasobeny cislom ktoreho rozsahy
# su dane absolutnou hodnotou v matici amp
# intenzita mutacie je v parametri rate
def mutm(old_pop, rate, amp, space):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape

    # kontrola ci faktor mutacie je v rozsahu 0 az 1
    rate = rate_check(rate)

    # vypocet poctu mutacii
    muts = math.floor(shape[0] * shape[1] * rate)

    # rozdiel v ohraniceniach (pomocne pre dalsie vypocty)
    amp_diff = np.subtract(amp[1], amp[0])

    # cyklus vyberu genu a jeho mutacia
    for i in range(muts):
        row = math.ceil(random.random() * shape[0]) - 1
        column = math.ceil(random.random() * shape[1]) - 1

        # mutacia cez pomocnu funkciu, ktora kontroluje ci je nova hodnota v rozsahu space
        old_pop[row, column] = mut_check(old_pop[row, column] * (random.random(
        ) * amp_diff[column] + amp[0, column]), space[0, column], space[1, column])


# vymena genov, permutacna mutacia
# funkcia vymeni 2 geny v jedincovi
# intenzita mutacie je v parametri rate
def swapgen(old_pop, rate):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape

    # kontrola ci faktor mutacie je v rozsahu 0 az 1
    rate = rate_check(rate)

    # vypocet poctu mutacii
    mutations = math.ceil(shape[0] * shape[1] * rate)

    # cyklus vyberu genu a jeho mutacia
    for i in range(mutations):
        r = math.ceil(random.random() * shape[0]) - 1

        # kontrola aby sa nevybral ten isty gen na vymenu
        c1 = 0
        c2 = 0
        while c1 == c2:
            c1 = math.ceil(random.random() * shape[1]) - 1
            c2 = math.ceil(random.random() * shape[1]) - 1

        old_pop[r, [c1, c2]] = old_pop[r, [c2, c1]]


# vymena dvoch casti jedincov, permutacna mutacia
# funkcia rozdeli nahodne jedince na 2 casti a
# vymeni ich poradie
# intenzita mutacie je v parametri rate
def swappart(old_pop, rate):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape

    # kontrola ci faktor mutacie je v rozsahu 0 az 1
    rate = rate_check(rate)

    # vypocet poctu mutacii
    mutations = math.ceil(shape[0] * rate)

    # cyklus vyberu genu a jeho mutacia
    for i in range(mutations):
        r = math.ceil(random.random() * shape[0]) - 1
        c = math.ceil(random.random() * shape[1]) - 1
        old_pop[r] = np.concatenate((old_pop[r][c:], old_pop[r][0:c]))


# ------------------------------ KRIZENIA ------------------------------

# medzilahle krizenie
# funkcia zkrizi pary jedincov cez vzorec a vytvori 2 jedince
def intmedx(old_pop, alpha, mode):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape

    # vytvorenie listu indexov dvojic, zamiesanie ak mode==0
    pair_list = list(range(shape[0]))
    if mode == 0:
        random.shuffle(pair_list)

    # hlavny cyklus krizenia, krizi sa len parny pocet retazcov
    # pri neparnom pocte ostane jeden povodny
    for cyk in range(int(shape[0] / 2)):

        i, j = pair_list[2 * cyk], pair_list[2 * cyk + 1]

        for g in range(shape[1]):

            dx = abs(old_pop[i, g] - old_pop[j, g])

            # od vacsieho genu sa odcitava, ku mensiemu pripocitava
            if old_pop[i, g] < old_pop[j, g]:
                old_pop[i, g] += random.random() * alpha * dx
                old_pop[j, g] -= random.random() * alpha * dx
            else:
                old_pop[i, g] -= random.random() * alpha * dx
                old_pop[j, g] += random.random() * alpha * dx


# krizenie medzi viacerymi rodicmi
# funkcia vytvori nove jedince, ktore bude mat
# geny skombinovane zo vsetkych jedincov v populacii
def crossgrp(old_pop, n):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape
    new_pop = []

    # hlavny cyklus krizenia
    for row in range(n):

        tmp = []

        # vyber genov nahodne z celej populacie
        for col in range(shape[1]):

            m = math.ceil(random.random() * shape[0]) - 1
            tmp.append(old_pop[m, col])

        new_pop.append(tmp)

    return np.array(new_pop)


# krizenie s vyberom poctu bodov krizenia
# funkcia zkrizi pary jedincov tak, ze sa podla parametra pts
# vytvoria body krizenia medzi ktorymi sa vymenia geny jedincov
def crossov(old_pop, pts, mode):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = old_pop.shape

    # vytvorenie listu indexov dvojic, zamiesanie ak mode==0
    pair_list = list(range(shape[0]))
    if mode == 0:
        random.shuffle(pair_list)

    # hlavny cyklus krizenia
    for cyk in range(int(shape[0] / 2)):
        i, j = pair_list[2 * cyk], pair_list[2 * cyk + 1]

        # vytvorenie listu bodov krizenia, na zaciatku musi byt 0
        # a na konci posledna pozicia retazcov
        split_points_list = [
            x + 1 for x in sorted(random.sample(range(shape[1] - 2), pts))]
        split_points_list.insert(0, 0)
        split_points_list.append(shape[1])

        # pomocna funkcia ktora vrati 2 pary s prehodenymi
        # genmi podla listu bodov krizenia
        old_pop[i], old_pop[j] = splitting(old_pop, split_points_list, i, j, 1)


# ------------------------------ ine funkcie ------------------------------

# generovanie novej populacie
def genrpop(pop_size, space):

    # shape[0] - pocet riadkov
    # shape[1] - pocet stlpcov
    shape = space.shape
    new_pop = []

    for row in range(pop_size):
        lst = []

        for column in range(shape[1]):
            limit = space[1][column] - space[0][column]
            lst.append(random.random() * limit + space[0][column])
            lst = pop_check(lst, column, space)

        new_pop.append(lst)

    return np.array(new_pop)


# generovanie populacie pre permutacne ulohy
# vygenerovanie hodnoty majo rozsah <1; max_val>
def genrpop_perm(rows, max_val):

    pop = []
    for i in range(rows):
        lst = list(range(max_val))
        random.shuffle(lst)
        pop.append(lst)

    return np.array(pop)


# inverzia fit hodnot pre maximalizacne ulohy
def invfit(old):

    new = (max(old) - old) + min(old)

    return new


# schwefel funkcia pre testovanie
def schwefel(pop):

    shape = pop.shape
    fit = np.zeros(shape[0])

    for i in range(shape[0]):
        fit[i] -= np.sum(pop[i] * np.sin(np.sqrt(np.abs(pop[i]))))

    return fit


# eggholder funkcia pre testovanie
def eggholder(pop):

    shape = pop.shape
    fit = np.zeros(shape[0])

    for i in range(shape[0]):
        x = pop[i]
        for j in range(shape[1] - 1):
            fit[i] -= x[j] * math.sin(math.sqrt(abs(x[j] - (x[j + 1] + 47)))) + (
                x[j + 1] + 47) * math.sin(math.sqrt(abs(x[j + 1] + 47 + x[j] / 2)))

    return fit


# rastrigin funkcia pre testovanie
def rastrigin(pop):

    shape = pop.shape

    fit = np.ones(shape[0]) * 10 * shape[1]

    pop_array = np.array(pop)

    for i in range(shape[0]):
        fit[i] += np.sum(pop_array[i]**2 - 10 *
                         np.cos(2 * np.pi * pop_array[i]))

    return fit


# ------------------------------ pomocne funkcie ------------------------------

# pomocna funkcia na vypis matic
def output(pop):
    for tmp in pop:
        print(str(["{0:0.2f}".format(i) for i in tmp]).replace("'", ""))
    print()


# pomocna funkcia na kontrolu hodnoty faktora mutacie
def rate_check(rate):
    if rate > 1:
        rate = 1

    elif rate < 0:
        rate = 0

    return rate


# pomocna funkcia na kontrolu hodnoty vygenerovani novej pop
def pop_check(pop, column, space):
    if pop[column] < space[0][column]:
        pop[column] = space[0][column]

    if pop[column] > space[1][column]:
        pop[column] = space[1][column]

    return pop


# pomocna funkcia na kontrolu hodnoty po mutacii
def mut_check(d, space_l, space_u):

    if d < space_l:
        d = space_l

    elif d > space_u:
        d = space_u

    return d


# pomocna funkcia pre crossov na rozdelenie listu
def splitting(old_pop, pts_list, i, j, mode):
    counter = 0
    new_row1 = np.array([])
    new_row2 = np.array([])

    for cyk in range(len(pts_list) - 1):
        # ak sa jedna o prvy riadok v cykle v crossov, poradie i a j ostava, takze sa zacina i-ckom
        if mode == 1:
            # vymena i a j pri kazdej iteracii
            if counter % 2 == 0:
                row_index1 = i
                row_index2 = j
            else:
                row_index1 = j
                row_index2 = i
        # ak sa jedna o druhy riadok v cykle v crossov, poradie i a j sa vymeni, takze sa zacina j-ckom
        elif mode == 2:
            # vymena i a j pri kazdej iteracii
            if counter % 2 == 0:
                row_index1 = j
                row_index2 = i
            else:
                row_index1 = i
                row_index2 = j

        counter += 1

        new_row1 = np.append(
            new_row1, old_pop[row_index1, pts_list[cyk]:pts_list[cyk + 1]])
        new_row2 = np.append(
            new_row2, old_pop[row_index2, pts_list[cyk]:pts_list[cyk + 1]])

    return new_row1, new_row2
