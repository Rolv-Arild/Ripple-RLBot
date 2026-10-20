"""RLBot v5 packet fixtures using the real SDK types."""
from rlbot import flat
from obs import BOOST_LOCATIONS


def field_info():
    return flat.FieldInfo(boost_pads=[flat.BoostPad(
        location=flat.Vector3(*loc), is_full_boost=bool(loc[2] > 72)) for loc in BOOST_LOCATIONS],
        goals=[], tiles=[])


def physics(x=0., y=0., z=17.):
    return flat.Physics(location=flat.Vector3(x,y,z), rotation=flat.Rotator(),
                        velocity=flat.Vector3(), angular_velocity=flat.Vector3())


def packet(frame, phase=flat.MatchPhase.Active, count=6, scores=(0, 0)):
    return flat.GamePacket(
        players=[flat.PlayerInfo(physics=physics(1000. * (-1)**i, -3000. + 900.*i),
            score_info=flat.ScoreInfo(), hitbox=flat.BoxShape(118.00738,84.19941,36.159073),
            hitbox_offset=flat.Vector3(13.87566,0.,20.754988),
            air_state=flat.AirState.OnGround, dodge_timeout=-1., demolished_timeout=-1.,
            name=f'Player {i}', team=i%2, boost=33., player_id=10+i,
            last_input=flat.ControllerState(), dodge_dir=flat.Vector2()) for i in range(count)],
        balls=[flat.BallInfo(physics=physics(0.,0.,93.))],
        boost_pads=[flat.BoostPadState(is_active=True, timer=0.) for _ in BOOST_LOCATIONS],
        teams=[flat.TeamInfo(team_index=i, score=score) for i, score in enumerate(scores)],
        match_info=flat.MatchInfo(frame_num=frame, seconds_elapsed=frame/120.,
            game_time_remaining=300., match_phase=phase, world_gravity_z=-650., game_speed=1.),
        tiles=[])
