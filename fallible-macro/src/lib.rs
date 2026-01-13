use syn::{parse_macro_input, ItemFn};
use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_attribute]
pub fn fallible(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let sig = &input.sig;
    let block = &input.block;
    let vis = &input.vis;

    let fn_name = sig.ident.to_string();
    let id_hash = fxhash::hash32(&fn_name.as_bytes());

    let expanded = quote! {
        #vis #sig {
            #[allow(unreachable_code)]
            if cfg!(feature = "fallible-sim") {
                return Err(
                    ::fallible::fallible_core::simulated_failure(
                        ::fallible::fallible_core::FailurePoint { id: ::fallible::fallible_core::FailurePointId(#id_hash) }
                    )
                );
            }

            #block
        }
    };

    expanded.into()
}