export type UserProfile = {
  email: string;
  id: string;
  name: string;
  role: string;
};

export type LoginPayload = {
  email: string;
  password: string;
};

export type SignupPayload = {
  confirm_password: string;
  email: string;
  name: string;
  password: string;
};
