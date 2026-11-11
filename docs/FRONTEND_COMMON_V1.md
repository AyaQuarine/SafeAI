# Interface SafeAI — cohérence avec Common V1.0

## Ce qui est livré

- `/` : **diagnostic d'acquisition** existant (WAV, VAD, qualité, microphone en mode natif).
- `/common` : **visualiseur des JSON V1** (`AlignedSession` et `DiarizationResult`), conçu pour la démonstration devant l'équipe.
- Exemples intégrés **fictifs** : conversation complète, silence confirmé et conversation partielle avec chevauchement.
- Choix de deux JSON réels depuis le navigateur, **sans requête d'upload**. Les labels `unknown`, `overlap` et les diagnostics sont affichés.
- Vérification client légère : même `session_id`, même durée (tolérance de 20 ms), horodatages dans la session.
- Chronologie des tours de Community-1 ; une **intersection entre deux locuteurs** n'est pas comptée comme deux événements simplement parce que deux tours sont marqués `overlap`.

## Ce qui n'est pas livré

La page ne lance pas Whisper, Community-1 ni l'alignement sur un WAV. Pour cela, utiliser la **CLI** et ouvrir ensuite les JSON produits. Aucun résultat de risque, d'émotion ou de diagnostic de santé n'est affiché.

## Confidentialité

Les JSON choisis par l'utilisateur sont lus avec `File.text()` par son navigateur ; les exemples fictifs sont servis comme fichiers statiques. L'interface ne doit pas être exposée sur Internet. Les WAV et transcriptions réelles demeurent sous `data/local_dev/` hors Git.

## Démarrage

```powershell
.\.venv\Scripts\python.exe -m safeai.acquisition ui
# ouvrir http://127.0.0.1:8765/common
```

La route est ajoutée dans `safeai.ui.app.create_app()`. Les tests FastAPI sont dans `tests/ui/test_common_dashboard.py`. Cette interface n'est pas un frontend de production ni un mode live.
