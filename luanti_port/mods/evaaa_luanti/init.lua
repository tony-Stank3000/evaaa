-- EVAAA Luanti bridge (experimental).
-- Communicates only with the local Python bridge over Luanti's async HTTP API.
-- This is not a validated reproduction of the original Unity environment.

local MOD_NAME = minetest.get_current_modname()
local http = minetest.request_http_api()
local BASE_URL = "http://127.0.0.1:8766"
local PROTOCOL_VERSION = 1
local POLL_INTERVAL = 0.20
local ACTION_TICKS = 5
local TURN_RADIANS = math.rad(15)
local SPEED = 2.5

local elapsed = 0
local get_in_flight = false
local post_in_flight = false
local current_player_name = nil
local food = 1.0
local water = 1.0
local episode_steps = 0

if not http then
  minetest.log("error", "[evaaa_luanti] HTTP API unavailable. Add secure.http_mods = " .. MOD_NAME .. " to minetest.conf.")
  return
end

local function log(msg)
  minetest.log("action", "[evaaa_luanti] " .. msg)
end

local function json_post(path, payload, callback)
  if post_in_flight then return end
  post_in_flight = true
  http.fetch({
    url = BASE_URL .. path,
    method = "POST",
    data = minetest.write_json(payload),
    extra_headers = { "Content-Type: application/json" },
    timeout = 1.0,
  }, function(result)
    post_in_flight = false
    if callback then callback(result) end
  end)
end

local function json_get(path, callback)
  if get_in_flight then return end
  get_in_flight = true
  http.fetch({url = BASE_URL .. path, method = "GET", timeout = 1.0}, function(result)
    request_in_flight = false
    if callback then callback(result) end
  end)
end

local function active_player()
  if current_player_name then
    local p = minetest.get_player_by_name(current_player_name)
    if p then return p end
  end
  local players = minetest.get_connected_players()
  if #players > 0 then
    current_player_name = players[1]:get_player_name()
    return players[1]
  end
  return nil
end

local function node_at(pos)
  local n = minetest.get_node_or_nil(pos)
  if not n then return "unknown" end
  return n.name or "unknown"
end

local function is_obstacle(name)
  if name == "air" or name == "ignore" then return false end
  local def = minetest.registered_nodes[name]
  return not (def and def.walkable == false)
end

local function build_observation(player)
  local pos = player:get_pos()
  local rgb = {}
  local cells = {}
  for z = -8, 7 do
    cells[z] = {}
    for x = -8, 7 do
      local p = {x = math.floor(pos.x + x), y = math.floor(pos.y), z = math.floor(pos.z + z)}
      cells[z][x] = node_at(p)
    end
  end

  -- 16x16 node-name raster expanded to 64x64. Synthetic encoding, not a screenshot.
  for py = 1, 64 do
    local gz = math.floor((py - 1) / 4) - 8
    local row = {}
    for px = 1, 64 do
      local gx = math.floor((px - 1) / 4) - 8
      local name = (cells[gz] and cells[gz][gx]) or "unknown"
      local id = minetest.get_content_id(name) or 0
      row[#row + 1] = {id % 256, math.floor(id / 256) % 256, (id * 31) % 256}
    end
    rgb[#rgb + 1] = row
  end

  local olfactory, thermo, collision = {}, {}, {}
  local dirs = {{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-2},{0,2}}
  for i, d in ipairs(dirs) do
    local sample = {x = math.floor(pos.x + d[1]), y = math.floor(pos.y), z = math.floor(pos.z + d[2])}
    local name = node_at(sample)
    local id = minetest.get_content_id(name)
    olfactory[i] = (id % 1000) / 1000.0
    collision[i] = is_obstacle(name) and 1.0 or 0.0
    if i <= 8 then
      local def = minetest.registered_nodes[name]
      thermo[i] = (def and def.liquidtype and def.liquidtype ~= "none") and 1.0 or 0.0
    end
  end
  local hp = player:get_hp() or 0
  local vel = player:get_velocity() or {y = 0}
  local touch = {math.abs(vel.y or 0) < 0.2 and 1.0 or 0.0}
  return {
    rgb = rgb, ev = {food, water, 0.0, hp / 20.0},
    olfactory = olfactory, thermo = thermo, collision = collision, touch = touch
  }
end

local function publish_observation()
  local player = active_player()
  if not player then return end
  food = math.max(0, food - 0.0005)
  water = math.max(0, water - 0.0007)
  episode_steps = episode_steps + 1
  local reward = -math.sqrt((1 - food)^2 + (1 - water)^2)
  json_post("/api/observation", {
    protocol_version = PROTOCOL_VERSION,
    player = player:get_player_name(),
    observation = build_observation(player),
    reward = reward,
    terminated = food <= 0 or water <= 0 or (player:get_hp() or 0) <= 0,
    truncated = episode_steps >= 10000,
    info = {episode_steps = episode_steps, simulator = "luanti", mapping = "experimental"}
  })
end

local function apply_action(player, action)
  local vel = player:get_velocity() or {x = 0, y = 0, z = 0}
  if action == 0 then
    player:set_velocity({x = 0, y = vel.y or 0, z = 0})
  elseif action == 1 then
    local yaw = player:get_look_horizontal()
    player:set_velocity({x = -math.sin(yaw) * SPEED, y = vel.y or 0, z = math.cos(yaw) * SPEED})
  elseif action == 2 then
    player:set_look_horizontal(player:get_look_horizontal() + TURN_RADIANS)
  elseif action == 3 then
    player:set_look_horizontal(player:get_look_horizontal() - TURN_RADIANS)
  elseif action == 4 then
    local inv = player:get_inventory()
    for _, stack in ipairs(inv:get_list("main") or {}) do
      if not stack:is_empty() then
        local def = minetest.registered_items[stack:get_name()]
        if def and def.groups and (def.groups.food or 0) > 0 then
          inv:remove_item("main", ItemStack(stack:get_name() .. " 1"))
          food = math.min(1.0, food + 0.25)
          break
        end
      end
    end
  end
end

local function reset_episode(player)
  food, water, episode_steps = 1.0, 1.0, 0
  local spawn = minetest.setting_get_pos("static_spawnpoint") or {x = 0, y = 1, z = 0}
  player:set_pos(spawn)
  player:set_velocity({x = 0, y = 0, z = 0})
end

local function poll_action()
  json_get("/api/next", function(result)
    if not result or not result.succeeded or not result.data then return end
    local msg = minetest.parse_json(result.data)
    if type(msg) ~= "table" or msg.protocol_version ~= PROTOCOL_VERSION then return end
    local player = active_player()
    if not player then return end
    if msg.type == "reset" then
      reset_episode(player)
      publish_observation()
    elseif msg.type == "step" and type(msg.action) == "number" and msg.action >= 0 and msg.action <= 4 then
      local action = math.floor(msg.action)
      apply_action(player, action)
      minetest.after(ACTION_TICKS * 0.05, function()
        -- Stop horizontal movement at the end of each action window.
        if action == 1 and player:is_player() then
          local vel = player:get_velocity() or {x = 0, y = 0, z = 0}
          player:set_velocity({x = 0, y = vel.y or 0, z = 0})
        end
        publish_observation()
      end)
    end
  end)
end

minetest.register_globalstep(function(dtime)
  elapsed = elapsed + dtime
  if elapsed >= POLL_INTERVAL then
    elapsed = 0
    poll_action()
  end
end)

minetest.register_on_joinplayer(function(player)
  if not current_player_name then current_player_name = player:get_player_name() end
  log("connected player " .. player:get_player_name())
end)

log("loaded; waiting for local Python bridge at " .. BASE_URL)
