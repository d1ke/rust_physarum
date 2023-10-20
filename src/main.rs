use bevy::{
    prelude::*, 
    render::render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
    window::{CursorGrabMode, PresentMode, WindowLevel, WindowTheme},
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
};
use rand::prelude::*;

#[derive(Resource)]
struct Settings {
    map_size: UVec2,
    agents_count: usize,
    agents_move_speed: f32,
    agents_rotation_speed: f32,
    sensors_distance: f32,
    sensors_angle: f32,
    pheromone_production_per_second: f32,
    pheromone_spread_coeff: f32,
    pheromone_dryout_per_second: f32,
}

#[derive(Resource)]
struct PheromoneMap {
    values: Vec<f32>,
    spread_buffer: Vec<f32>,
}

#[derive(Component)]
struct Position {
    value: Vec2,
    previous_frame_value: Vec2,
}

#[derive(Component)]
struct Velocity {
    value: Vec2,
}

#[derive(Component)]
struct Agent;

#[derive(Component)]
struct MapSprite;

/// A group of related system sets, used for controlling the order of systems. Systems can be
/// added to any number of sets.
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
enum SystemSet {
    BeforeRender,
    Logic,
    Render,
    AfterRender,
}

// const FIXED_DELTA_TIME: f32 = 0.02;

fn main() {
    App::new()
    .add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Physarum".into(),
                resolution: (1500., 1500.).into(),
                present_mode: PresentMode::AutoNoVsync,
                // Tells wasm to resize the window according to the available canvas
                // fit_canvas_to_parent: true,
                // Tells wasm not to override default event handling, like F5, Ctrl+R etc.
                // prevent_default_event_handling: false,
                window_theme: Some(WindowTheme::Dark),
                // This will spawn an invisible window
                // The window will be made visible in the make_visible() system after 3 frames.
                // This is useful when you want to avoid the white window that shows up before the GPU is ready to render the app.
                // visible: false,
                ..default()
            }),
            ..default()
        }),
        LogDiagnosticsPlugin::default(),
        FrameTimeDiagnosticsPlugin,
    ))
        .add_systems(Startup, startup_system)
        // .insert_resource(FixedTime::new_from_secs(FIXED_DELTA_TIME))
        // .configure_set(
        //     FixedUpdate,
        //     SystemSet::Logic
        // )
        .configure_sets(
            Update, 
            (SystemSet::BeforeRender, SystemSet::Logic, SystemSet::Render, SystemSet::AfterRender).chain())
        .add_systems(
            Update,
            (
                (agent_rotation_system, agent_movement_system, pheromone_production_system, pheromone_spreading_system, pheromone_dryout_system).chain().in_set(SystemSet::Logic),
                (clear_map_texture_system).in_set(SystemSet::BeforeRender),
                (draw_pheromone_system).chain().in_set(SystemSet::Render),//draw_agents_system
            ),
        )
        .run();
}

fn startup_system(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    let settings = Settings {
        map_size: UVec2 {x: 1024, y: 1024} * 2,
        agents_count: 50000,
        agents_move_speed: 100.0,
        agents_rotation_speed: 30.0,
        sensors_angle: 30.0,
        sensors_distance: 5.0,
        pheromone_production_per_second: 99999.0,
        pheromone_spread_coeff: 2.0,
        pheromone_dryout_per_second: 0.1,
    };

    // Agents
    let map_size_float = settings.map_size.as_vec2();
    for _ in 0..settings.agents_count {
        let position = Vec2 { x: random::<f32>() * map_size_float.x, y: random::<f32>() * map_size_float.y };

        commands.spawn((
            Agent,
            Position { value: position, previous_frame_value: position },
            Velocity { value: Vec2 { x: random::<f32>() - 0.5, y: random::<f32>() - 0.5 }.normalize() * settings.agents_move_speed },
        ));
    }

    // Camera
    commands.spawn(Camera2dBundle::default());

    // Map
    let size = Extent3d {
        width: settings.map_size.x,
        height: settings.map_size.y,
        ..default()
    };

    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };

    image.resize(size);

    let image_handle = images.add(image);

    commands.spawn((SpriteBundle {
        texture: image_handle,
        transform: Transform::from_scale(Vec3::ONE * 0.5),
        ..default()
    }, MapSprite));

    // Resources
    commands.insert_resource(PheromoneMap { 
        values: vec![0.0; (settings.map_size.x * settings.map_size.y) as usize],
        spread_buffer: vec![0.0; (settings.map_size.x * settings.map_size.y) as usize],
    });
    commands.insert_resource(settings);
}

fn agent_rotation_system(
    settings: Res<Settings>,
    time: Res<Time>,
    pheromone_map: Res<PheromoneMap>,
    mut query: Query<(&Position, &mut Velocity), With<Agent>>,
) {
    let map_size = settings.map_size.as_vec2();

    let left_sensor_rotation = Vec2::from_angle(-settings.sensors_angle.to_radians()); // todo bake
    let right_sensor_rotation = Vec2::from_angle(settings.sensors_angle.to_radians());
    let left_rotation_per_frame = Vec2::from_angle(-settings.agents_rotation_speed.to_radians() * time.delta_seconds());
    let right_rotation_per_frame = Vec2::from_angle(settings.agents_rotation_speed.to_radians() * time.delta_seconds());

    let left_rotation_per_frame = left_sensor_rotation;
    let right_rotation_per_frame = right_sensor_rotation;

    for (position, mut velocity) in &mut query {
        let middle_sensor_offset = velocity.value.normalize() * settings.sensors_distance;
        
        let middle_sensor_position = (position.value + middle_sensor_offset).rem_euclid(map_size);
        let left_sensor_position = (position.value + left_sensor_rotation.rotate(middle_sensor_offset)).rem_euclid(map_size);
        let right_sensor_position = (position.value + right_sensor_rotation.rotate(middle_sensor_offset)).rem_euclid(map_size);

        let middle_sensor_value = pheromone_map.values[to_pixel_index(middle_sensor_position, settings.map_size)];
        let left_sensor_value = pheromone_map.values[to_pixel_index(left_sensor_position, settings.map_size)];
        let right_sensor_value = pheromone_map.values[to_pixel_index(right_sensor_position, settings.map_size)];

        if left_sensor_value > middle_sensor_value && middle_sensor_value > right_sensor_value {
            velocity.value = left_rotation_per_frame.rotate(velocity.value);
        }
        else if right_sensor_value > middle_sensor_value && middle_sensor_value > left_sensor_value {
            velocity.value = right_rotation_per_frame.rotate(velocity.value);
        }
        else if left_sensor_value > middle_sensor_value && right_sensor_value > middle_sensor_value {
            let rotation = if random::<bool>() { left_rotation_per_frame } else { right_rotation_per_frame };
            velocity.value = rotation.rotate(velocity.value);
        }
    }
}

fn agent_movement_system(
    settings: Res<Settings>,
    time: Res<Time>,
    mut query: Query<(&mut Position, &Velocity), With<Agent>>
) {
    let map_size = settings.map_size.as_vec2();

    for (mut position, velocity) in &mut query {
        position.previous_frame_value = position.value;
        position.value = (position.value + velocity.value * time.delta_seconds()).rem_euclid(map_size);
    }
}

fn pheromone_production_system(
    settings: Res<Settings>,
    time: Res<Time>,
    mut pheromone_map: ResMut<PheromoneMap>,
    positions: Query<&Position, With<Agent>>
) {
    let increment = settings.pheromone_production_per_second * time.delta_seconds();

    for position in &positions {
        let pixel_index = to_pixel_index(position.value, settings.map_size);
        pheromone_map.values[pixel_index] = (pheromone_map.values[pixel_index] + increment).min(1.0);

        // let delta_position = position.value.as_ivec2() - position.previous_frame_value.as_ivec2();
        // let steps_count = delta_position.x.abs().max(delta_position.y.abs()) + 1;
        // let steps_count_f = steps_count as f32;
        // let mut pos = position.previous_frame_value;

        // for _ in 0..steps_count {
        //     pos.x += delta_position.x as f32 / steps_count_f;
        //     pos.y += delta_position.y as f32 / steps_count_f;

        //     let pixel_index = to_pixel_index(pos, settings.map_size);
        //     pheromone_map.values[pixel_index] = (pheromone_map.values[pixel_index] + increment).min(1.0);
        // }
    }
}

fn pheromone_spreading_system(
    settings: Res<Settings>,
    time: Res<Time>,
    mut pheromone_map: ResMut<PheromoneMap>,
) {
    let map_length = pheromone_map.values.len();
    let mut prev_neighbour_index = map_length - 1;
    let mut next_neighbour_index = 1;

    let neighbour_coeff = (settings.pheromone_spread_coeff * time.delta_seconds()).min(0.5);
    let center_coeff = 1.0 - neighbour_coeff * 2.0;

    for current_index in 0..map_length {
        pheromone_map.spread_buffer[current_index] =
            pheromone_map.values[current_index] * center_coeff +
            pheromone_map.values[prev_neighbour_index] * neighbour_coeff +
            pheromone_map.values[next_neighbour_index] * neighbour_coeff;

        prev_neighbour_index = (prev_neighbour_index + 1) % map_length;
        next_neighbour_index = (next_neighbour_index + 1) % map_length;
    }

    let mut prev_neighbour_index = ((settings.map_size.y - 1) * settings.map_size.x) as usize;
    let mut next_neighbour_index = settings.map_size.x as usize;

    for current_index in 0..map_length {
        pheromone_map.values[current_index] =
            pheromone_map.spread_buffer[current_index] * center_coeff + 
            pheromone_map.spread_buffer[prev_neighbour_index] * neighbour_coeff +
            pheromone_map.spread_buffer[next_neighbour_index] * neighbour_coeff;

        prev_neighbour_index = (prev_neighbour_index + 1) % map_length;
        next_neighbour_index = (next_neighbour_index + 1) % map_length;
    }
}

fn pheromone_dryout_system(
    settings: Res<Settings>,
    time: Res<Time>,
    mut pheromone_map: ResMut<PheromoneMap>,
) {
    let decrement = settings.pheromone_dryout_per_second * time.delta_seconds();

    for value in pheromone_map.values.iter_mut() {
        *value = (*value - decrement).max(0.0);
    }
}

fn clear_map_texture_system(
    mut images: ResMut<Assets<Image>>, 
    map_image_handle: Query<&Handle<Image>, With<MapSprite>>
) {
    if let Some(map_image) = images.get_mut(map_image_handle.single()) {
        map_image.data.fill(0);
    }
}

fn draw_pheromone_system(
    mut images: ResMut<Assets<Image>>, 
    pheromone_map: Res<PheromoneMap>,
    map_image_handle: Query<&Handle<Image>, With<MapSprite>>
) {
    if let Some(map_image) = images.get_mut(map_image_handle.single()) {
        let mut pixel_index = 0;
        for x in pheromone_map.values.iter() {
            map_image.data[pixel_index + 1] = (x * 255.0) as u8;
            map_image.data[pixel_index + 3] = (x * 255.0) as u8;
            pixel_index += 4;
        }
    }
}

fn draw_agents_system(
    settings: Res<Settings>,
    mut images: ResMut<Assets<Image>>, 
    map_image_handle: Query<(&Handle<Image>), With<MapSprite>>,
    positions: Query<&Position, With<Agent>>
) {
    if let Some(map_image) = images.get_mut(map_image_handle.single()) {
        for position in &positions {
            let pixel_index = to_pixel_index(position.value, settings.map_size) * 4;
            map_image.data[pixel_index + 0] = 255;
            map_image.data[pixel_index + 3] = 255;
        }
    }
}

fn to_pixel_index(vec: Vec2, map_size: UVec2) -> usize {
    ((vec.y as u32).min(map_size.y - 1) * map_size.x + (vec.x as u32).min(map_size.x - 1)) as usize
}
