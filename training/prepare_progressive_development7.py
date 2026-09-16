"""Freeze a fresh development screen after certificate-003, without launching games."""
import copy
import progressive_campaign as campaign


def main():
    state = campaign.load(campaign.OUT / 'state.json')
    assert state['active_baseline'] == 'proxy-v2-512-lowprior'
    assert not state.get('pending_certificate')
    assert len(state['accepted_upgrades']) == 2
    assert state['certificate_attempts'][-1]['attempt'] == 3
    assert state['certificate_attempts'][-1]['passed'] is False
    reference = state['accepted_upgrades'][-1]['policy']
    common = dict(
        engine='torch', device='cuda', opponents='actual-proxy',
        proxy_directory='artifacts/progressive-upgrades/opponent-proxies-v2',
        worlds=2048, prior_weight=.003, bullhead_weight=.002,
        continuation_by_players={
            '4': 'artifacts/progressive-upgrades/ppo-specialist-4-v1/model.pt'},
        player_overrides={'4': dict(worlds=2048, prior_weight=.003,
                                   bullhead_weight=.002, prior_from_continuation=True)},
    )
    configs = {}
    pair = copy.deepcopy(common)
    pair.update(worlds=256, pair_worlds=512, pair_topk=3)
    pair['player_overrides']['4']['pair_worlds'] = 0
    configs['pair512-specialist2048'] = pair
    pure = copy.deepcopy(common)
    pure.update(prior_weight=0., bullhead_weight=0.)
    configs['purewin2048-specialist2048'] = pure
    belief = copy.deepcopy(common)
    belief.update(belief_oversample=4, belief_history=3, belief_strength=.35)
    belief['player_overrides']['4']['belief_oversample'] = 0
    configs['belief2048-specialist2048'] = belief
    distill = copy.deepcopy(common)
    distill['continuation_by_players']['3'] = (
        'artifacts/progressive-upgrades/continuation-v2/model.pt')
    configs['distill2048-specialist2048'] = distill
    path = campaign.OUT / 'development-007/manifest.json'
    assert not path.exists(), 'Do not replace an already frozen experiment'
    path = campaign.prepare_run('development-007', configs, 256, 'development', reference)
    manifest = campaign.load(path)
    manifest['selection_protocol'] = dict(
        screen_deals_per_player_count=256,
        eligibility='Observed gain of at least 0.5 percentage point in BOTH modes.',
        ranking='Descending minimum of the two paired gains, then descending mean gain, then lexicographic policy ID.',
        shortlist_limit=2,
        next_stage='Freeze a fresh development run with 512 deals per player count, the same accepted reference and the exact shortlisted policies.',
        finalist_eligibility='Both observed gains at least 0.5 percentage point AND both ordinary paired 95% bootstrap lower bounds greater than zero in that complete 512-deal development run.',
        finalist_ranking='Descending minimum ordinary paired 95% bootstrap lower bound, then descending minimum gain, then mean gain, then lexicographic policy ID.',
        if_none_eligible='Continue development; do not alter this run or promote a noneligible candidate.',
        no_interim_selection=True,
        formal_certificate='A selected finalist still requires a fresh original-protocol certificate, with the next alpha allocation; development results never count as upgrades.',
    )
    manifest['development_only_rationale'] = (
        'Compare four existing independent development hypotheses: two-card planning, '
        'pure win-share objective, public-history belief and a newly distilled continuation. '
        'Each uses a 2048-world four-player specialist; three-player settings differ. '
        'No certificate traces enter training. All policies use ordinary Torch execution, '
        'without CUDA graphs. The failed third certificate and its alpha allocation remain recorded.')
    manifest['preflight_protocol'] = dict(
        script='training/verify_progressive_config_service.py',
        script_sha256=campaign.sha(campaign.ROOT / 'training/verify_progressive_config_service.py'),
        archive='development-006', archive_policy='reference',
        observations='All ten decisions in all rotations of the first complete archived block at each player count.',
        expected='Serial direct calls to each frozen archived Planner; check all resulting cards are legal.',
        transport='The same observations and seeds must return exactly the same cards through the shared service.',
        streams=2, clients=8,
    )
    campaign.dump(path, manifest)
    print(path, flush=True)


if __name__ == '__main__':
    main()
