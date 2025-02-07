function example_rename_events(evts)
    evts = rename(evts, :condition => :animal)
    evts.animal[evts.animal.=="car"] .= "dog"
    evts.animal[evts.animal.=="face"] .= "cat"
    evts = rename(evts, :continuous => :eye_movement_size)
    return evts
end
