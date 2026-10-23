"""Create a focused named W&B workspace while preserving the prior workspace."""
import copy
import datetime
import json
from pathlib import Path
import uuid

import wandb


def main():
    cfg = json.loads(Path('induced_coding_language/config.json').read_text())
    run_id = cfg['run_id']
    out = Path('runs')/run_id/'archive/workspace'
    out.mkdir(parents=True,exist_ok=True)
    api = wandb.Api(api_key=Path('/workspace/gpu-awareness-private/wandb-personal-api-key').read_text().strip(),
                    overrides={'base_url':'https://api.wandb.ai'})
    cloud = api.run('iarcuschin/gpu-awareness/'+run_id)
    assert cloud.id == run_id
    query = '''query View($entityName:String,$name:String,$viewType:String,$viewName:String){
    project(name:$name,entityName:$entityName){allViews(viewType:$viewType,viewName:$viewName){
    edges{node{id name type displayName spec}}}}}'''

    def read(name):
        result = api._service_api.execute_graphql(query,variables={
            'entityName':'iarcuschin','name':'gpu-awareness','viewType':'project-view','viewName':name})
        return [edge['node'] for edge in result['project']['allViews']['edges']]

    previous = read('nw-nwuseriarcuschin-w')
    assert len(previous) == 1
    previous = previous[0]
    (out/'previous-experiment-workspace.json').write_text(json.dumps(previous,indent=2)+'\n')
    name = 'nw-induced-coding-language'
    existing = read(name)
    assert not existing, 'Named workspace already exists; inspect it before updating.'
    spec = copy.deepcopy(json.loads(previous['spec']))
    section = spec['section']
    bank = section['panelBankConfig']
    wanted = ['train','eval','selection','full_eval','interval_eval']
    groups = [g for g in bank['sections'] if g['name'] in wanted]
    bank['sections'] = groups
    assert [g['name'] for g in groups] == ['train','eval','selection','full_eval','interval_eval']
    for group in groups:
        group['__id__'] = uuid.uuid4().hex[:9]
        group['isPanelsAuto'] = False
        for panel in group['panels']:
            panel['__id__'] = uuid.uuid4().hex[:9]
            panel['config']['chartTitle'] = panel['config']['chartTitle'].replace('solution-token','first-token')
            assert panel['config']['xAxis'] == 'update'
    for runset in section['runSets']:
        runset['search'] = {'query':'^'+run_id+'$','isRegex':True}
        runset['selections'] = {'root':0,'bounds':[],'tree':[run_id]}
    panels = [p for g in groups for p in g['panels']]
    assert len(panels) == 11 and bank['settings']['hideSearchQuery'] == ''
    assert section['workspaceSettings']['shouldAutoGeneratePanels'] is False
    assert not any(m.startswith(('judge_progress/','baseline_eval/','selected_eval/','generation/'))
                   for p in panels for m in p['config']['metrics'])
    mutation = '''mutation UpsertView2($id:ID,$entityName:String,$projectName:String,$type:String,
    $name:String,$displayName:String,$spec:String){upsertView(input:{id:$id,entityName:$entityName,
    projectName:$projectName,name:$name,displayName:$displayName,type:$type,spec:$spec,createdUsing:WANDB_SDK}){
    view{id name} inserted}}'''
    response = api._service_api.execute_graphql(mutation,variables={
        'id':None,'entityName':'iarcuschin','projectName':'gpu-awareness','type':previous['type'],
        'name':name,'displayName':run_id,'spec':json.dumps(spec)})
    assert response['upsertView']['inserted']
    saved = read(name)
    assert len(saved) == 1 and json.loads(saved[0]['spec']) == spec
    assert read(previous['name'])[0]['spec'] == previous['spec']
    (out/'workspace.json').write_text(json.dumps(saved[0],indent=2)+'\n')
    receipt = dict(run_id=run_id,verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        workspace_id=saved[0]['id'],workspace_name=name,sections=[g['name'] for g in groups],
        panels=11,x_axis='update',previous_workspace_unchanged=True,hidden_panels=0,
        persisted_spec_readback_verified=True,browser_interaction=False)
    (out.parent.parent/'workspace-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
