// mod/VirtualInput.cs
using InControl;

namespace HKRLBot
{
    public struct ActionButtons
    {
        public bool Left, Right, Up, Down, Jump, Attack, Dash;
    }

    public class VirtualDevice : InputDevice
    {
        public ActionButtons State;

        public VirtualDevice() : base("HKRL Virtual Device")
        {
            AddControl(InputControlType.DPadLeft, "Left");
            AddControl(InputControlType.DPadRight, "Right");
            AddControl(InputControlType.DPadUp, "Up");
            AddControl(InputControlType.DPadDown, "Down");
            AddControl(InputControlType.Action1, "Jump");
            AddControl(InputControlType.Action3, "Attack");
            AddControl(InputControlType.RightTrigger, "Dash");
        }

        public override void Update(ulong updateTick, float deltaTime)
        {
            UpdateWithState(InputControlType.DPadLeft, State.Left, updateTick, deltaTime);
            UpdateWithState(InputControlType.DPadRight, State.Right, updateTick, deltaTime);
            UpdateWithState(InputControlType.DPadUp, State.Up, updateTick, deltaTime);
            UpdateWithState(InputControlType.DPadDown, State.Down, updateTick, deltaTime);
            UpdateWithState(InputControlType.Action1, State.Jump, updateTick, deltaTime);
            UpdateWithState(InputControlType.Action3, State.Attack, updateTick, deltaTime);
            UpdateWithState(InputControlType.RightTrigger, State.Dash, updateTick, deltaTime);
            Commit(updateTick, deltaTime);
        }
    }

    public class VirtualInput
    {
        public VirtualDevice Device = new VirtualDevice();
        private bool attached;

        public void Attach()
        {
            if (attached) return;
            // GameManager and its InputHandler exist from the title menu
            // onward, but Attach can be called before either is up (the boot
            // path calls it every macro tick); returning without setting
            // `attached` keeps the call retryable until they exist.
            if (GameManager.instance == null || GameManager.instance.inputHandler == null) return;
            InputManager.AttachDevice(Device);
            var a = GameManager.instance.inputHandler.inputActions;
            a.left.AddBinding(new DeviceBindingSource(InputControlType.DPadLeft));
            a.right.AddBinding(new DeviceBindingSource(InputControlType.DPadRight));
            a.up.AddBinding(new DeviceBindingSource(InputControlType.DPadUp));
            a.down.AddBinding(new DeviceBindingSource(InputControlType.DPadDown));
            a.jump.AddBinding(new DeviceBindingSource(InputControlType.Action1));
            a.attack.AddBinding(new DeviceBindingSource(InputControlType.Action3));
            a.dash.AddBinding(new DeviceBindingSource(InputControlType.RightTrigger));
            // Menus (the title screen, the save-profile select, the statue's
            // challenge menu) read menuSubmit, not jump. Bound to the same
            // control as Jump so every confirm pulse in ResetMacro doubles as
            // a UI submit -- this is what lets a freshly booted game walk
            // itself through the main menu into the save.
            a.menuSubmit.AddBinding(new DeviceBindingSource(InputControlType.Action1));
            attached = true;
        }

        public void Apply(ActionButtons b) => Device.State = b;
        public void Clear() => Device.State = new ActionButtons();
    }
}
