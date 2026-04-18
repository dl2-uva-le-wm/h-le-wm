# PushT Hi-LEWM: spiegazione dettagliata (training vs inferenza)

Questo documento spiega il flusso **attuale del codice** nel repo per il caso PushT, con esempi numerici concreti.

## 1) Config reale usata in training (P2)

Valori da `/config/train/hi_lewm.yaml` e `/config/train/data/hi_pusht.yaml`:

- `training.train_low_level: False`
- `wm.history_size: 3`
- `wm.high_level.waypoints.num: 5`
- `wm.high_level.waypoints.strategy: random_sorted`
- `wm.high_level.waypoints.min_stride: 1`
- `wm.high_level.waypoints.max_span: 15`
- `data.dataset.num_steps: history_size + max_span = 3 + 15 = 18`
- `data.dataset.frameskip: 5`
- `loss.alpha: 0.0`
- `loss.beta: 1.0`
- `loss.sigreg.weight: 0.0`
- `loader.batch_size: 128`
- `optimizer.lr: 5e-5`

Note tecniche:

- `wm.action_dim` non e' hardcoded in YAML: viene inferito dal dataset a runtime (`dataset.get_dim("action")`).
- L'input azione effettivo ai moduli che codificano azioni e':
  - `effective_act_dim = frameskip * wm.action_dim`.

## 2) Cosa significa `history_size=3` qui

Nel training P2 attuale, `history_size=3` serve soprattutto a:

- fissare il primo waypoint a `t1 = history_size - 1 = 2`;
- determinare `num_steps = history_size + max_span = 18`;
- mantenere compatibilita' con il ramo P1.

Importante:

- In **P2** non viene fatto frame-stacking locale `[t-2, t-1, t]` per ogni waypoint.
- In **P1** c'e' una sequenza di contesto lunga `history_size` nel predictor low-level.

## 3) Training P2: passo per passo con esempio numerico

Questa sezione descrive cosa accade in `hi_lejepa_forward`.

### 3.1 Batch in ingresso

Per un batch:

- `pixels`: `(B, T, C, H, W)` con `T=18`
- `action`: `(B, T, Aeff)` dove `Aeff = frameskip * action_dim`

Con config corrente: `B=128`, `T=18`.

### 3.2 Sampling waypoints

Si campionano `N=5` waypoints ordinati.

- Primo fissato: `w1 = 2`
- Ultimo vincolo: `w5 - w1 <= max_span = 15`
- Tutti entro la finestra `0..17`

Esempio valido (uno tra i tanti possibili):

- `w = [2, 5, 9, 13, 16]`
- gap: `[3, 4, 4, 3]`
- span totale: `16 - 2 = 14 <= 15`

### 3.3 Encoding visivo (fix #1 fast path)

Con `train_low_level=False` e `sigreg.weight=0`, si usa fast path:

- non codifica tutti i 18 frame;
- codifica solo i 5 frame waypoint.

Quindi:

- prima: costo encoder su `T=18` frame/sample
- ora: costo encoder su `N=5` frame/sample

Riduzione teorica forward encoder: circa `18/5 = 3.6x` meno frame.

### 3.4 Costruzione `z_context` e `z_target`

Dopo `z_waypoints` (shape `(B, 5, D)`):

- `z_context = z_waypoints[:, :-1]` -> `(B, 4, D)`
- `z_target = z_waypoints[:, 1:]` -> `(B, 4, D)`

Con l'esempio:

- `z_context = [z(w1), z(w2), z(w3), z(w4)]`
- `z_target  = [z(w2), z(w3), z(w4), z(w5)]`

Ogni posizione e' una transizione consecutiva.

### 3.5 Chunk azioni per transizione waypoint (fix #2 vettorizzato)

Intervalli:

- `starts = [w1, w2, w3, w4]`
- `ends   = [w2, w3, w4, w5]`

Nell'esempio:

- intervalli: `[2:5], [5:9], [9:13], [13:16]`
- lunghezze: `[3, 4, 4, 3]`
- `Lmax = 4`

Build vettorizzato:

- `chunk_actions`: `(B, K, Lmax, Aeff)` con `K=N-1=4`
- `chunk_mask`: `(B, K, Lmax)`
- flatten a `(B*K, Lmax, Aeff)`, un'unica chiamata al `latent_action_encoder`
- output `macro_actions`: `(B, K, Dl)` cioe' `(B, 4, Dl)`

### 3.6 Input reale al predittore high-level

Input a `predict_high`:

- `emb = z_context` shape `(B, 4, D)`
- `macro_actions` shape `(B, 4, Dl)`

Poi:

- `macro_actions -> macro_to_condition -> high_cond (B, 4, D)`
- `high_predictor(emb, high_cond) -> z_pred (B, 4, D)`

Loss principale:

- `l2_pred_loss = mean((z_pred - z_target)^2)`
- dato che `alpha=0`, `beta=1`, `sigreg.weight=0`, la loss totale coincide con `l2_pred_loss`.

## 4) Perche' P2 non usa "[t-2,t-1,t] per ogni waypoint"

Nel design attuale:

- P1 modella dinamica locale fine (usa contesto temporale corto nel predictor low-level).
- P2 modella salti tra waypoint: usa
  - token latente al waypoint,
  - macro-action del tratto tra waypoint.

Quindi in P2 l'informazione del "mezzo" tra due waypoint passa soprattutto dalla macro-action, non da frame stacking locale attorno al waypoint.

## 5) Inferenza/planning gerarchico: flusso reale

Nel policy gerarchico:

1. Si codifica **solo l'ultimo frame** osservato per stato iniziale `z_init`.
2. Si codifica **solo l'ultimo frame** del goal per `z_goal`.
3. Planner high-level (CEM) propone azioni macro candidate.
4. `rollout_high(z_init, latent_actions)` simula traiettorie latenti high-level.
5. Si prende un subgoal latente.
6. Planner low-level pianifica azioni primitive verso quel subgoal.

### 5.1 Punto chiave: puo' partire con un solo waypoint in input

Si': in `rollout_high`, storia iniziale:

- `z_hist` parte con lunghezza 1.
- al primo passo il contesto e' `ctx=1`.

Quindi il predittore high-level viene chiamato davvero anche con sequenza lunga 1.

## 6) "Predice tutto insieme o uno alla volta?"

Dipende dal contesto:

- **Training P2**: predice in parallelo tutta la sequenza `(N-1)` in una forward.
- **Planning rollout_high**: genera autoregressivamente passo passo nel loop.

Questa differenza e' normale: training con teacher-forcing, planning con self-rollout.

## 7) FAQ rapide

### 7.1 "Se in planning parte da 1 token, in training l'ha mai visto?"

Si', in training con attenzione causale la prima posizione vede solo il primo token.

### 7.2 "Allora e' sicuramente perfetto?"

No: anche se non e' un bug logico, resta possibile il classico gap teacher-forcing vs rollout autoregressivo (exposure bias).

### 7.3 "Perche' allora history_size=3 nel P2?"

Nel tuo codice attuale, principalmente:

- eredita design P1,
- impone ancoraggio iniziale (`w1=2`),
- definisce insieme a `max_span` la lunghezza finestra `num_steps=18`.

Non implica automaticamente frame stacking locale dei waypoint in P2.

## 8) Esempio completo compatto (numeri)

Config:

- `history_size=3`, `max_span=15`, `num_steps=18`, `N=5`.

Sample:

- `w=[2,5,9,13,16]`.

Costruzioni:

- `z_waypoints`: `(B,5,D)`
- `z_context`: `(B,4,D)` = waypoint 1..4
- `z_target`: `(B,4,D)` = waypoint 2..5
- `macro_actions`: `(B,4,Dl)` da chunk azioni sui segmenti:
  - `[2:5]`, `[5:9]`, `[9:13]`, `[13:16]`
- `z_pred`: `(B,4,D)`
- `loss`: MSE tra `z_pred` e `z_target`.

Questo e' il cuore del training P2 attuale.
